'''
Pipeline Input:
1. Tune RF model on feature selection datasets
2. Use tuned model in RFE are refit on data
3. Unsupervised selection of important features via RFECV-RF
4. Create new training dataset (.transform) are prepare training dataset
5. Create new test set using the features determined by the pipeline

Pipeline Output:
- optimum number of features
- ranking of features it list format [rank, feature]
-  input training dataset as pandas with optimum no. of features
- input test dataset as pandas with optimum no. of features
- target variable training dataset
- target variable test dataset
'''

def pipeline_feature_selection(fs_data, training_data, test_data, calibration_data, verbose):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint as sp_randint
    from sklearn.model_selection import GroupKFold
    from sklearn.feature_selection import RFECV
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    feature_to_drop = ['Discharge_Q', 'SOH_discharge_capacity', 'Group']
    feature_to_predict = 'Discharge_Q'

    X_train = fs_data.drop(feature_to_drop, axis=1)
    y_train = fs_data[feature_to_predict]

    no_of_features = 1  # number of features to drop after each iteration

    # Hyper-param for Random Forest
    # 10*len(list(X_train))
    # used to have boostrap = 500
    rf_tuning = RandomForestRegressor(n_estimators=500, bootstrap=True, n_jobs=-1) #full_dataset: 700 estimators, 500 boostraps
                                                                                 #standford_dataset: 70 estimators
    # note: oxford uses: 250 est with 1- iter
    #       standford uses: 250 est with 50 iter
    param = {"max_depth": sp_randint(15, 25), #15-25, 5-10
             "max_features": [no_of_features], #[no_of_features],  # sp_randint(2, 4),
             "min_samples_split": sp_randint(2, 5),
             "min_samples_leaf": sp_randint(5, 15),
             "criterion": ['mse']}

    # no_top_models = 5
    no_of_splits = len(np.unique(fs_data.Group))  # number of slits is equal to the number of groups
    groups = fs_data.Group
    group_kfold = GroupKFold(n_splits=no_of_splits)  # inner test and train using the group KFold

    model = RandomizedSearchCV(rf_tuning, param_distributions=param, cv=group_kfold, n_iter=100, # full_dataset: 150
                               iid=False, refit=True, verbose=verbose)
    model.fit(X_train, y_train, groups=groups)
    RF_f_selection_model = model.best_estimator_
    # RF_f_selection_model_param = model.best_params_

    # '''Recurrent Feature Elimination'''
    names = list(fs_data.drop(['Discharge_Q', 'SOH_discharge_capacity', 'Group'], axis=1))

    rf = RF_f_selection_model
    rfe = RFECV(estimator=rf, min_features_to_select=no_of_features, cv=group_kfold, step=1,
                scoring='neg_mean_squared_error', verbose=verbose)  # neg_mean_squared_error, r2

    # selector_RF = rfe.fit(X_train_scaled, y_train)
    selector_RF = rfe.fit(X_train, y_train, groups=groups)

    ranking_features = sorted(zip(map(lambda x: round(x, 4), selector_RF.ranking_), names), reverse=False)
    optimumum_no_feature = selector_RF.n_features_

    x = range(no_of_features, len(selector_RF.grid_scores_) + no_of_features)
    y = selector_RF.grid_scores_

    '''feature selection resuts'''
    print('Feature rank: \n {}'.format(ranking_features))
    # Plot number of features VS. cross-validation scores
    f = plt.figure(figsize=(7, 5))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(x, y, 'o--', color='tab:orange')
    plt.plot(x[np.argmax(y)], np.max(y), 'v', markersize=15, color='k')
    # plt.title('Optimum number of features based RF-RFE using neg-mse is: {}'.format(optimumum_no_feature))
    plt.xlabel('Selected no. of features', fontsize=15)
    plt.ylabel('Cross-validation score [Negative MSE]', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(False)
    plt.show()


    # transform dataset based on optimum features
    training_data_opt_fet_X = pd.DataFrame(selector_RF.transform(training_data.drop(feature_to_drop, axis=1))) # input feature space training as DataFrame
    test_data_opt_fet_X = pd.DataFrame(selector_RF.transform(test_data.drop(feature_to_drop, axis=1))) # input feature space testing as DataFrame
    calibration_data_opt_fet_X = pd.DataFrame(selector_RF.transform(calibration_data.drop(feature_to_drop, axis=1))) # input feature space testing as DataFrame

    '''Adding 'Group' feature to the dataset'''
    # add the group so that you can re-tune future models based on
    # training_data_opt_fet_X_new = pd.concat([training_data_opt_fet_X, training_data.Group], axis=1)
    training_data_opt_fet_X['Group'] = np.array(training_data.Group)
    # test_data_opt_fet_X_mew = pd.concat([test_data_opt_fet_X, test_data.Group], axis=1)
    test_data_opt_fet_X['Group'] = np.array(test_data.Group)

    # calibration_data_opt_fet_X_mew = pd.concat([calibration_data_opt_fet_X, calibration_data.Group], axis=1)
    calibration_data_opt_fet_X['Group'] = np.array(calibration_data.Group)


    train_y = training_data[feature_to_predict]
    test_y = test_data[feature_to_predict]
    calibration_y = calibration_data[feature_to_predict]

    return optimumum_no_feature, ranking_features, training_data_opt_fet_X, test_data_opt_fet_X, calibration_data_opt_fet_X, train_y, test_y, calibration_y, f
