
'''Functions used for plotting results'''
import numpy as np
import forestci as fci
import scipy.stats as st

def rmspe(y_true, y_pred):
    '''
    Compute Root Mean Square Perceantage Error between two arrays.
    '''
    loss = np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true))))
    return loss #output in %

def mape(y_true, y_pred):
    '''
    Compute Root Mean Square Perceantage Error between two arrays.
    '''
    loss = np.mean(np.abs((y_true - y_pred)) / y_true)

    return loss

def alpha_accuracy_beta_rlh(y, y_hat, std, alpha_accuracy):
    '''Function for uncertainty quantification'''

    error = y - y_hat
    N = len(y)

    # calculate the predictions failing in the alpha-accuracy zone
    acc_zone = []
    for idx, i in enumerate(error):
        if -alpha_accuracy*y[idx] <= i <= alpha_accuracy*y[idx]:
            acc_zone.append(i)
    no_of_entries_acc_zone = np.round(len(acc_zone) / N, 2)  # predictions between or on aboundry of accuracy zone

    # calculate Beta based on intersection between predicted variance and accuracy zone based on 2*STD CI else put 0
    z_l = -alpha_accuracy*y / std
    z_h = alpha_accuracy*y / std
    p_l = st.norm.cdf(z_l)
    p_h = st.norm.cdf(z_h)

    beta_prob = np.round(np.mean(abs(p_h - p_l)), 2)

    # calculate easrly and late estimates
    early_estimates = []
    late_estimates = []
    for idx, i in enumerate(error):
        # if -alpha_accuracy*y[idx] < i:
        if i>0:
            early_estimates.append(i)
        # elif i < alpha_accuracy*y[idx]:
        else:
            late_estimates.append(i)
    # ratio of late to early
    if len(late_estimates) == 0:
        r_lh = len(early_estimates)
    else:
        r_lh = np.round(len(early_estimates) / len(late_estimates), 2)  # ratio of late to early estimates

    # number of early estimates
    early_estimates_percentage = len(early_estimates)/N*100

    return no_of_entries_acc_zone, beta_prob, early_estimates_percentage


def count_entries_per_interval(y, y_hat, std_hat):
    # model relibility curve calculation
    # returns probability for the algorithm based on 'prob' varianle and associated interval

    # refrence for zscore: https://stackoverflow.com/questions/20864847/probability-to-z-score-and-vice-versa-in-python
    # reference paper: https://arxiv.org/abs/1807.00263

    # statisical z-tabels inherent to regression
    no_of_probs = len(y)  # len(y), 21 or 11
    prob = np.linspace(0, 1, no_of_probs) # no. of prob considered
    import scipy.stats as st
    z_score = []  # z value according to above prop

    for i in prob:
        z_score.append(st.norm.ppf(1 - (1 - i) / 2))

    n = len(y)  # number of entries
    bins_list_y_in_interval = [[] for _ in range(no_of_probs)]

    # calculation per bin based on z-score
    y = np.array(y)
    y_hat = np.array(y_hat)
    # calculate entries per bin
    for idx, i in enumerate(y):
        for j in range(len(bins_list_y_in_interval)):
            if y_hat[idx] - z_score[j] * std_hat[idx] <= i <= y_hat[idx] + z_score[j] * std_hat[idx]:
                bins_list_y_in_interval[j].append(i)

    prob_per_interval = []
    for k in bins_list_y_in_interval:
        if not k:
            prob_per_interval.append(0)
        else:
            prob_per_interval.append(len(k) / n)

    # associate a probability for each prediction based on the bin that first entered
    prob_y = []
    for idx_k, k in enumerate(y):
        for l in range(len(bins_list_y_in_interval)):
            if y_hat[idx_k] - z_score[l] * std_hat[idx_k] <= k <= y_hat[idx_k] + z_score[l] * std_hat[idx_k]:
                prob_y.append(prob_per_interval[l])
                break

    prob_y_expected = []
    for idx_k, k in enumerate(y):
        for l in range(len(bins_list_y_in_interval)):
            if y_hat[idx_k] - z_score[l] * std_hat[idx_k] <= k <= y_hat[idx_k] + z_score[l] * std_hat[idx_k]:
                prob_y_expected.append(prob[l])
                break

    # calculate probability per interval when compared to the total no. of sampels in the data

    return prob_per_interval, prob_y, prob_y_expected, prob  # prob_y is the prob calculated from interval and appears only once in each interval
                                                            # prob_per_interval is the observed confidence level


# model probability
def predict_prob(y, y_hat, std_hat):  # calculate predicted probability form the model using the normal distribution cdf
    #    calculate probability of an output based on CDF
    z = abs(np.array(y) - y_hat) / std_hat
    a = st.norm.cdf(z)
    b = st.norm.cdf(-z)
    prob_model = a - b

    return prob_model


# calculate the new variance to accomodate for calibration
def std_calibrated(y, y_hat, prob_model_calibrated):
    Z = st.norm.ppf(1 - (1 - prob_model_calibrated) / 2)
    # if Z = 0 means that the error is far too big hence asign a big prob
    for idx, p in enumerate(prob_model_calibrated):
        if p>=0.999:
            Z[idx] = 1.7 #force Z to a big number since this will be infinity
        elif p <=0.01:
            Z[idx] = 0.1 # force Z to a small number such that STD predicted is small


    std_cal = abs(np.array(y) - y_hat) / Z

    return std_cal

def calibration_isotonic_regression_model(model_name, model, X_calibration, y_calibration, X_train):
    # 1. function that trains the calibration regressor using as input calibration data in the first instance
    # 2. it then takes in the prob_out of the mdel on the test and outputs calibrated prob for further calculation of
    # calibrated std
    # ref: https: // arxiv.org / abs / 1807.00263
    if model_name in ['Bayes_Ridge_model', 'GPR_model']:
        y_hat_calibration, sem_hat_calibration = model.predict(X_calibration, return_std=True)

    elif model_name == 'RF_model':
        y_hat_calibration = model.predict(X_calibration)
        sem_hat_calibration = np.sqrt(fci.random_forest_error(model, X_train, X_calibration))

    else:
        print('Error: Not able to calculate variace!')
        # y_hat, sem = model.predict(X_calibration)

    prob_per_int_y_calibration, prob_y_calibration, prob_y_calibration_expected, prob = count_entries_per_interval(
        y_calibration, y_hat_calibration, sem_hat_calibration)
    prob_model_y_calibration = predict_prob(y_calibration, y_hat_calibration, sem_hat_calibration)

    # isotonic regression
    from sklearn.isotonic import IsotonicRegression as IR
    ir = IR(out_of_bounds='clip')
    ir.fit(prob_model_y_calibration, prob_y_calibration)

    return ir

def calibrated_prob(model_calibration, prob_model_to_be_calibrated):
    prob_test_calibrated = model_calibration.transform(prob_model_to_be_calibrated)
    return prob_test_calibrated


# function used in NN

def calibration_isotonic_regression(model_name, model, prob_model, X_calibration, y_calibration, X_train):
    # 1. function that trains the calibration regressor using as input calibration data in the first instance
    # 2. it then takes in the prob_out of the mdel on the test and outputs calibrated prob for further calculation of
    # calibrated std
    # ref: https: // arxiv.org / abs / 1807.00263
    if model_name == 'Bayes_Ridge_model':
        y_hat_calibration, sem_hat_calibration = model.predict(X_calibration, return_std=True)

    elif model_name == 'RF_model':
        y_hat_calibration = model.predict(X_calibration)
        sem_hat_calibration = np.sqrt(fci.random_forest_error(model, X_train, X_calibration))

    else:
        print('Error: Not able to calculate variace!')
        # y_hat, sem = model.predict(X_calibration)

    prob_per_int_y_calibration, prob_y_calibration, prob_y_calibration_expected, prob = count_entries_per_interval(
        y_calibration, y_hat_calibration, sem_hat_calibration)
    prob_model_y_calibration = predict_prob(y_calibration, y_hat_calibration, sem_hat_calibration)

    # isotonic regression
    from sklearn.isotonic import IsotonicRegression as IR
    ir = IR(out_of_bounds='clip')
    ir.fit(prob_model_y_calibration, prob_y_calibration)

    prob_test_calibrated = ir.transform(prob_model)
    return prob_test_calibrated
