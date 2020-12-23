
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint, expon
from sklearn.model_selection import GroupKFold, KFold

'''Random Forest'''

path = [] # specify data path

# specify battery specific feature_selection and test datasets

battery_list =  [] # select battery group

'''Do you want adversarial case?'''
adversarial = True


for i in battery_list:

    battery = i
    # load relevant battery dataset for training the algorithm
    if adversarial:
        data_train = pd.read_csv(path + 'data_train_fsed__adversarial' + battery + '.csv', index_col=0)
    else:
        data_train = pd.read_csv(path + 'data_train_fsed_' + battery + '.csv', index_col=0)

    X_train = data_train.drop(['Discharge_Q', 'Group'], axis=1)
    y_train = data_train['Discharge_Q']

    '''Hyper-param for Random Forest'''
    no_of_DT_estimators =1500 # feature selection is using 700

    rf = RandomForestRegressor(n_estimators=no_of_DT_estimators, bootstrap=True, n_jobs=-1)

    # make sure the max_feature in RF param does not exceed no. of input features
    if len(list(X_train)) < 7:
        max_no_of_fet = len(list(X_train))
    else:
        max_no_of_fet = 7

    param = {
             "max_depth": sp_randint(10, 25),
             "max_features": sp_randint(3, max_no_of_fet),
             "min_samples_split": sp_randint(3, 9),
             "min_samples_leaf": sp_randint(5, 15),
             "criterion": ['mse']} # ['mse', 'friedman_mse', 'mae']

    groups = data_train.Group
    no_of_splits = len(np.unique(groups))  # number of slits is equal to the number of groups
    group_kfold = GroupKFold(n_splits=no_of_splits)
    model = RandomizedSearchCV(rf, param_distributions=param, cv=group_kfold, n_iter=20,
                               iid=False, verbose=2)
    # fit model
    model.fit(X_train, y_train, groups=groups)

    # save the model to disk
    if adversarial:
        filename = 'RF_model_adversarial_' + battery + '.sav'
    else:
        filename = 'RF_model_' + battery + '.sav'

    path_for_model = [] # save trained model
    pickle.dump(model, open(path_for_model + filename, 'wb'))
