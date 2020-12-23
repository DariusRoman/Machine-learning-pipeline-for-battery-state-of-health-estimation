import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as st
from operator import itemgetter


scaler = StandardScaler() # scaling function

'''Deep Ensamble'''

path = [] # specify training data path

# import functions for accuracy assesment

folder = [] # path to plotting
import sys
sys.path.append(folder)
from results_plotting_functions import rmspe, mape, alpha_accuracy_beta_rlh, count_entries_per_interval, predict_prob, std_calibrated

'''Do you want adversarial case?'''
adversarial = True

# train data
if adversarial:
    data_train = pd.read_csv(path + 'data_train_fsed__adversarial' + battery + '.csv', index_col=0)
else:
    data_train = pd.read_csv(path + 'data_train_fsed_' + battery + '.csv', index_col=0)

train_x = data_train.drop(['Discharge_Q', 'Group'], axis=1)
train_y = np.array(data_train.Discharge_Q).reshape(len(data_train), 1)

# calibration data
data_calibration = pd.read_csv(path + 'data_calibration_fsed_' + battery +'.csv', index_col=0)
data_calibration.reset_index(inplace=True, drop=True)

# test data
data_test = data_train[data_train.Group==data_train.Group.unique()[-1]] # test the model on the last battery group for visualisation purpuses
test_x = data_test.drop(['Discharge_Q', 'Group'], axis=1)
test_y = np.array(data_test.Discharge_Q).reshape(len(data_test), 1)

# training data
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# data for the NN input - column names
column_names = list(data_train.drop(['Group'], axis=1))
num_rows = len(data_train)
num_columns = len(column_names)
num_data = num_columns - 1

# Parameters of training
Learning_rate = 0.35e-3
epsilon = 1e-10 # epsilon choice explanation: https://stackoverflow.com/questions/43221065/how-does-the-epsilon-hyperparameter-affect-tf-train-adamoptimizer

num_iter = 17500
batch_size = 2500

test_ratio = 0.1
gpu_fraction = 0.9

# Ensemble NN - len of NN results in the number of NNs in the ensemble
NN = ['Dense_NN1', 'Dense_NN2', 'Dense_NN3']

print("Train data shape: " + str(train_x.shape))
print("Test data shape: " + str(test_x.shape))

neurons_layer_1 = int(1.5*len(list(test_x[0])))
neurons_layer_2 = int(len(list(test_x[0]))/2)
neurons_layer_3 = int(neurons_layer_2/2)

# Dense [input size, output size]
dense0 = [num_data, neurons_layer_2]
dense3 = [neurons_layer_2, neurons_layer_3]
dense_mu  = [neurons_layer_3, 1]
dense_var = [neurons_layer_3, 1]

#-----------------------------------------------------------------------------------------------------------------------
# create functions for each of the variabels

# initialise weight
def weight_variable(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

# initialise bias
def bias_variable(name, shape):
    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

# reference for the initialization of the weights based on xavier's paper see link below for tf explanation and paper:
# https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer


# create the network
def network_create(network_name):
    input_x = tf.placeholder(tf.float64, shape = [None, num_data], name = 'x-input')

    with tf.variable_scope(network_name):
        # Densely connect layer variables
        w_0 = weight_variable(network_name + '_w_0', dense0)
        b_0 = bias_variable(network_name + '_b_0', [dense0[1]])

        w_3 = weight_variable(network_name + '_w_1', dense3)
        b_3 = bias_variable(network_name + '_b_1', [dense3[1]])

        w_mu = weight_variable(network_name + '_w_mu', dense_mu)
        b_mu = bias_variable(network_name + '_b_mu', [dense_mu[1]])

        w_var = weight_variable(network_name + '_w_var', dense_var)
        b_var = bias_variable(network_name + '_b_var', [dense_var[1]])

    fc0 = tf.nn.relu(tf.matmul(input_x, w_0) + b_0)
    fc3 = tf.nn.leaky_relu(tf.matmul(fc0, w_3) + b_3)

    # output layer
    output_mu = tf.matmul(fc3, w_mu) + b_mu
    outputvar = tf.matmul(fc3, w_var) + b_var
    output_var_pos = tf.log(1 + tf.exp(output_var)) + 1e-06

    y = tf.placeholder(tf.float64, shape=[None, 1], name='y-output')

    # Negative Log Likelihood(NLL) cost function
    loss = tf.reduce_mean(0.5 * tf.log(output_var_pos) + 0.5 * tf.div(tf.square(y - output_mu), output_var_pos)) + 1e-06

    # Variables - for GRAPH see: https://www.tensorflow.org/api_docs/python/tf/Graph#get_collection
    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, network_name)

    # Gradient clipping -  options: GradientDescent, Adagrad and Adadelta
    optimizer = tf.train.AdamOptimizer(learning_rate=Learning_rate, epsilon=epsilon, use_locking=False)

    comp_grad = optimizer.compute_gradients(loss, var_list=vars)
    capped_comp_grad = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in comp_grad]
    train_opt = optimizer.apply_gradients(capped_comp_grad)

    return input_x, y, output_mu, output_var_pos, loss, train_opt, vars


# Make batch data
def making_batch(data_size, sample_size, data_x, data_y):

    batch_idx = np.random.choice(data_size, sample_size)

    batch_x = np.zeros([sample_size, num_data])
    batch_y = np.zeros([sample_size, 1])

    for i in range(batch_idx.shape[0]):
        batch_x[i, :] = data_x[batch_idx[i], :]
        batch_y[i, :] = data_y[batch_idx[i], :]

    return batch_x, batch_y

#
# #-----------------------------------------------------------------------------------------------------------------------
# # Initialize Ensemble NN

x_list = []
y_list = []
output_mu_list = []
output_var_list = []
loss_list = []
train_list = []
train_var_list = []
output_test_list = []

# Train each ensemble network
for i in range(len(NN)):
    x_input, y, output_mu, output_var, loss, train_opt, train_vars = network_create(NN[i])

    x_list.append(x_input)
    y_list.append(y)
    output_mu_list.append(output_mu)
    output_var_list.append(output_var)
    loss_list.append(loss)
    train_list.append(train_opt)
    train_var_list.append(train_vars)

#-----------------------------------------------------------------------------------------------------------------------
# Create Session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

#-----------------------------------------------------------------------------------------------------------------------
# Set parameters for printing and testing
# Set parameters for printing and testing
num_print = 100
test_size = 500 # oxford_dataset: 100

len_train = len(train_x) #train_x.shape[0]
len_test = len(test_x) #test_x.shape[0]

loss_train = np.zeros([len(NN)])
out_mu = np.zeros([test_size, len(NN)])
out_var = np.zeros([test_size, len(NN)])

loss_matrix = [] # store the loss in a list so that you can plot the curve at the end
for iter in range(num_iter):
    # Making batches(testing)
    batch_x_test, batch_y_test = making_batch(len_test, test_size, test_x, test_y)
    # batch_x_test, batch_y_test, test_size = making_batch(data_test)

    for i in range(len(NN)):
        # Making batches(training)
        batch_x, batch_y = making_batch(len_train, batch_size, train_x, train_y)

        # Training
        _, loss, mu, var = sess.run([train_list[i], loss_list[i], output_mu_list[i], output_var_list[i]],
                                    feed_dict={x_list[i]: batch_x, y_list[i]: batch_y})

        # Testing
        loss_test, mu_test, var_test = sess.run([loss_list[i], output_mu_list[i], output_var_list[i]],
                                                feed_dict={x_list[i]: batch_x_test, y_list[i]: batch_y_test})

        if np.any(np.isnan(loss)):
            raise ValueError('There is Nan in loss')

        loss_train[i] += loss
        out_mu[:, i] = np.reshape(mu_test, test_size)
        out_var[:, i] = np.reshape(var_test, test_size)

    # Get final test result
    out_mu_final = np.mean(out_mu, axis=1)
    out_var_final = np.sqrt(np.mean(out_var + np.square(out_mu), axis=1) - np.square(out_mu_final))

    if iter % num_print == 0 and iter != 0:
        print(('-------------------------') + ' Iteration: ' + str(iter) + ' -------------------------')
        print('Average Loss(NLL): ' + str(loss_train / num_print))
        print('mean: ' + str(out_mu[0, :]))
        print('standard deviation: ' + str(np.sqrt(out_var[0, :])))
        print('Final capacity(mu): ' + str(out_mu_final[0]))
        print('Final STD: ' + str(out_var_final[0]))
        print('True capacity: ' + str(batch_y_test[0]))
        print(('--------------------------------------------------------------------------------------')
        print('\n')

        loss_matrix.append(list(loss_train / num_print)[0])
        loss_train = np.zeros(len(NN))

# plot loss
plt.figure(figsize=(10, 10))
x = np.arange(100, (len(loss_matrix)+1)*100, 100)
plt.plot(x, loss_matrix, 'o-', color='red')
plt.ylabel('Loss')
plt.xlabel('Epoch no.')
plt.show()

# '''Plot the results of the model'''

def predict_w_DNN(data):
    test_x  = np.array(data.drop(['Discharge_Q', 'Group'], axis=1))
    test_y  = np.array(data['Discharge_Q']).reshape(len(data), 1)

    test_x =  scaler.transform(test_x)
    test_size = len(test_x)

    # Get Unknown dataset and test
    x_sample = test_x
    y_sample = test_y
    # output for ensemble network
    out_mu_sample = np.zeros([x_sample.shape[0], len(NN)])
    out_var_sample = np.zeros([x_sample.shape[0], len(NN)])

    # output for single network
    out_mu_single = np.zeros([x_sample.shape[0], 1])
    out_var_single = np.zeros([x_sample.shape[0], 1])

    for i in range(len(NN)):
        mu_sample, var_sample = sess.run([output_mu_list[i], output_var_list[i]],
                                         feed_dict={x_list[i]: x_sample})

        out_mu_sample[:, i] = np.reshape(mu_sample, (x_sample.shape[0]))
        out_var_sample[:, i] = np.reshape(var_sample, (x_sample.shape[0]))

        out_mu_single[:, 0] = np.reshape(mu_sample[:, 0], (x_sample.shape[0]))
        out_var_single[:, 0] = np.reshape(var_sample[:, 0], (x_sample.shape[0]))

    out_mu_sample_final = np.mean(out_mu_sample, axis=1)
    out_var_sample_final = np.sqrt(np.mean(out_var_sample + np.square(out_mu_sample), axis=1) - np.square(out_mu_sample_final))

    # # easier notation
    m = out_mu_sample_final
    varma = out_var_sample_final

    # convert to termonilogy used in all python files so far (redundant from a code perspective)
    y_true = y_sample.flatten()
    y_hat = m # predicted mean

    return y_true, y_hat, varma

def calibration_isotonic_regression(data_calibration, prob_model): # calibration function

    y_true_calibration, y_hat_calibration, sem_hat_calibration = predict_w_DNN(data_calibration)

    prob_per_int_y_calibration, prob_y_calibration, prob_y_calibration_expected, prob = count_entries_per_interval(y_true_calibration, y_hat_calibration, sem_hat_calibration)
    prob_model_y_calibration = predict_prob(y_true_calibration, y_hat_calibration, sem_hat_calibration)

    # isotonic regression
    from sklearn.isotonic import IsotonicRegression as IR
    ir = IR(out_of_bounds='clip')
    ir.fit(prob_model_y_calibration, prob_y_calibration)

    prob_test_calibrated = ir.transform(prob_model)
    return prob_test_calibrated


r2_vec = []
mape_vec = []
rmspe_vec = []
mse_vec = []
acc_zone_percentage_vec = []
beta_vec = []
rlh_vec = []
avg_calibration_vec = []
SH_vec = []

for gr in data_test.Group.unique():

    data = data_test[data_test.Group==gr]
    data.reset_index(inplace=True, drop=True)

    # calculate accuracy metrics from regression point of view
    y_true, y_hat, sem = predict_w_DNN(data)
    # accuracy zone calculation
    z_value = st.norm.ppf(1-(1-0.9)/2) # quantile corresponding to 90%
    y_up = y_hat + z_value * sem
    y_low = y_hat - z_value * sem

    #  calculate model probability
    prob_per_int_y_test, prob_y_test, prob_y_test_expected, prob = count_entries_per_interval(y_true, y_hat, sem)
    prob_model_y_test = predict_prob(y_true, y_hat, sem)

    # calibration

    # load relevant calibration data
    if battery == 'full_dataset':
        if gr in [33., 38.]:
            group_calibration = [37.]
        elif gr in [331., 381.]:
            group_calibration = [371.]
        elif gr in [13.]:
            group_calibration = [11.]
        elif gr in [6., 18., 25., 28., ]:
            group_calibration = [28.]
        elif gr in [117.,  118.,  115.,  116., 1111., 1112., 1115., 1116., 1123., 1124., 1127., 1128.]:
            group_calibration = [1111., 1120.]
        else:
            raise ValueError('Calibration dataset not loaded!')
        data_calibration_group = data_calibration[data_calibration.Group.isin(group_calibration)]

    else:
        data_calibration_group = data_calibration

#     load calibration data per group
    prob_test_calibrated = calibration_isotonic_regression(data_calibration_group, prob_model_y_test)

    sem_calibrated = std_calibrated(y_true, y_hat, prob_test_calibrated)
    prob_per_int_y_test_calibrated, prob_y_test_calibrated, prob_y_test_expected, prob = count_entries_per_interval(y_true, y_hat, sem_calibrated)

    # accuracy zone calculation after calibration
    y_up_calibrated = y_hat + z_value * sem_calibrated
    y_low_calibrated = y_hat - z_value * sem_calibrated

    # accuracy calculation
    rmspe_error = rmspe(y_true, y_hat)
    r2 = r2_score(y_true, y_hat)
    mape_error = np.mean(np.abs(y_true-y_hat)/y_true)
    mse = mean_squared_error(y_true, y_hat)
    # calculate predictions in accuracy zone
    alpha_acc_zone = 0.015
    acc_zone_percentage, beta, rlh = alpha_accuracy_beta_rlh(y_true, y_hat, sem_calibrated, alpha_acc_zone)
    CS = 1 / len(y_true) * np.sum((abs(y_hat - y_true) < z_value * sem))
    CS_calibrated = 1 / len(y_true) * np.sum((abs(y_hat - y_true) < z_value * sem_calibrated))
    SH = np.mean(sem_calibrated)
    # re-calibrate model
    fig1= plt.figure(figsize=(10,10))
    x_axis = list(data.index.values)
    plt.plot(x_axis, y_true, 'ro', markersize=4, alpha=.6, label='True capacity')
    plt.plot(x_axis, y_hat, 'o', color='k', markersize=4, alpha=.6, label='Predicted capacity')
    plt.fill_between(x_axis, y_low_calibrated, y_up_calibrated, color='orange', alpha=0.4, label='±2$\varma$')
    plt.legend(fontsize=20)
    plt.xlabel('Cycle', fontsize=20)
    if battery == 'oxford_dataset':
        plt.ylabel('Capacity [mAh]', fontsize=20)
    else:
        plt.ylabel('Capacity [Ah]', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.show()

    r2_vec.append(r2)
    mape_vec.append(mape_error)
    rmspe_vec.append(rmspe_error)
    mse_vec.append(mse)
    acc_zone_percentage_vec.append(acc_zone_percentage)
    beta_vec.append(beta)
    rlh_vec.append(rlh)
    avg_calibration_vec.append(CS_calibrated)
    SH_vec.append(SH)

    if len(prob) < 11:  # plot reliability curves for an r2 better greater than 0
        pass
    else:
        # note: if there is not suffiecint data in index prob rounding should go to 1
        index = []
        round_value = 2    # <---- for more granularity in the         | calibration curve increase these no.
        for q in np.round(np.linspace(0, 1, 11), round_value): #       |
            if gr == 1001.0:
                index.append(np.where(np.round(prob, round_value) == q)[0][2])  # was [0][1] i.e. checking second element, used to raound ro 2
            else:
                index.append(np.where(np.round(prob, round_value) == q)[0][0])

        prob_expected = itemgetter(*index)(prob)
        prob_observed_uncalibrated = itemgetter(*index)(prob_per_int_y_test)
        prob_observed_calibrated = itemgetter(*index)(prob_per_int_y_test_calibrated)
        fig3 = plt.figure(figsize=(10, 10))
        plt.title('Test data group {}'.format(gr))
        plt.plot(prob_expected, prob_expected, 'k--', markersize=2)
        plt.plot(prob_expected, prob_observed_uncalibrated, 'bo--', markersize=10, linewidth=6)
        plt.plot(prob_expected, prob_observed_calibrated, 'o--', c='orange', markersize=10, linewidth=6)
        plt.xlabel('Expected Cofidence Level', fontsize=20)
        plt.ylabel('Observed Confidence Level', fontsize=20)
        plt.legend(['Ideal calibration', 'Uncalibrated', 'Calibrated'], fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.show()
    #     plt.grid(True)
        plt.figure(figsize=(10, 10))
        plt.plot(abs(y_true - y_hat), sem_calibrated, '*')
        plt.ylabel('Variance')
        plt.xlabel('Error')
        plt.show()

        ''' Print the two capacities on the same plot without the cycle number'''
        fig2 = plt.figure(figsize=(10, 10))
        # acc zone display uniformly
        y = np.linspace(np.min(y_true), np.max(y_true), 100)
        y_up = y + alpha_acc_zone*y
        y_low = y - alpha_acc_zone*y
        ax1 = plt.axes()  # standard axes
        ax1.plot(y_true, y_true, 'ro', alpha=.7, zorder=10, label='True Capacity ($y^*$)', markersize=2)
        ax1.fill_between(y, y_low, y_up, color='green', alpha=0.6, zorder=10,
                         label='Accuracy zone [-$\\alpha$, $\\alpha$]')

        markers, caps, bars = ax1.errorbar(y_true, y_hat, yerr=z_value * sem_calibrated, ecolor='orange', capsize=2,
                                           capthick=2,
                                           fmt='--o', color='black', alpha=0.6, zorder=5,
                                           label='Predicted capacity ($\hat{y}^*$) ±2$\varma$')

        # loop through bars and caps and set the alpha value
        [bar.set_alpha(0.4) for bar in bars]
        [cap.set_alpha(0.4) for cap in caps]
        ax1.invert_xaxis()

        plt.xlabel('True Capacity [Ah]', fontsize=20)
        if battery == 'oxford_dataset':
            plt.ylabel('Capacity [mAh]', fontsize=20)
        else:
            plt.ylabel('Capacity [Ah]', fontsize=20)
        plt.legend(fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)

        ax2 = plt.axes([0.21, 0.21, 0.24, 0.24])

        percent_error = (-y_true[1:] + y_hat[1:] ) / y_true[1:]
        if battery in ['full_dataset', 'standford_dataset']:
            no_of_bins = 50
        else:
            no_of_bins = 10  # 50-full_datset, 10 all other datasets
        n, bins, patches = ax2.hist(percent_error, no_of_bins, edgecolor='white', color='black')

        for c, p in zip(bins, patches):
            if c >= alpha_acc_zone:
                plt.setp(p, 'facecolor', 'black')
            elif c <= -alpha_acc_zone:
                plt.setp(p, 'facecolor', 'black')
            else:
                plt.setp(p, 'facecolor', 'green')  # mediumslateblue
        plt.axvline(0, color='red', linestyle='dashed')  # , linewidth=1)
        plt.axvline(-alpha_acc_zone, color='green', linestyle='dashed')  # , linewidth=1)
        plt.axvline(alpha_acc_zone, color='green', linestyle='dashed')  # , linewidth=1)
        plt.axvline(0, color='red', linestyle='dashed')  # , linewidth=1)
        plt.xlabel('% error: $(\hat{y}^*-y^*)/y^*$', fontsize=15)
        plt.ylabel('No. of entries', fontsize=15)
        plt.title('Histogram of % error wrt $\hat{y}^*$', fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.show()

    print('\n ----------Performance cell no {}-----------'.format(gr))
    print('RMSPE: {}'.format(rmspe_error * 100))
    print('MAPE: {}'.format(mape_error * 100))
    print('Mean Mean Square Error: {}'.format(mse))
    print('Mean % of entriers falling in the accuracy zone: {}'.format(acc_zone_percentage * 100))
    print('Mean average probability mass of the prediciton PDF within the accuracy zone: {}'.format(beta))
    print('Mean ratio of early to late predictions when references to accurazy zone boundries: {}'.format(rlh))
    print('Mean calibration score calibrated model: {}'.format(CS_calibrated))
    print('Mean sharpness: {}'.format(SH))

print('\n -----------  Overall results -----------')
print('RMSPE: {}'.format(np.mean(rmspe_vec)*100))
print('MAPE: {}'.format(np.mean(mape_vec)*100))
print('Mean Mean Square Error: {}'.format(np.mean(mse_vec)))
print('Mean % of entriers falling in the accuracy zone: {}'.format(np.mean(acc_zone_percentage_vec)*100))
print('Mean average probability mass of the prediciton PDF within the accuracy zone: {}'.format(np.mean(beta_vec)))
print('Mean ratio of early to late predictions when references to accurazy zone boundries: {}'.format(np.mean(rlh_vec)))
print('Mean calibration score calibrated model: {}'.format(np.mean(avg_calibration_vec)))
print('Mean sharpness: {}'.format(np.mean(SH_vec)))
