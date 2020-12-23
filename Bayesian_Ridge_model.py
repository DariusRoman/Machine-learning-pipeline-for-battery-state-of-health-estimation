
import numpy as np
import pandas as pd
import pickle
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
# Scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler


'''Bayesian Ridge - the linear model'''

path = [] # specify data path

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


    '''Hyper-param tunning Bayesian Ridge Regressiomn'''

    pipeline = Pipeline(

                            [
                                ('scl', StandardScaler()),
                                ('clf', linear_model.BayesianRidge(normalize=False, fit_intercept=True))
                            ]
                        )
    # old parameter: np.round(np.random.uniform(-0.0000001, 30, 1000), 2)
    param = {"clf__alpha_1": np.round(np.random.uniform(-0.01, 1000, 100), 2),
             "clf__alpha_2": np.round(np.random.uniform(-0.01, 1000, 100), 2),
             "clf__lambda_1": np.round(np.random.uniform(-0.01, 1000, 100), 2),
             "clf__lambda_2": np.round(np.random.uniform(-0.01, 1000, 100), 2),
            }

    no_of_splits = len(np.unique(data_train.Group))  # number of slits is equal to the number of groups
    groups = data_train.Group
    group_kfold = GroupKFold(n_splits=no_of_splits)
    model = RandomizedSearchCV(pipeline, param_distributions=param, cv=group_kfold, n_iter=10,
                               iid=True, verbose=10)
    # fit model
    model.fit(X_train, y_train, groups=groups)


    # save the model to disk
    if adversarial:
        filename = 'Bayesian_Ridge_adversarial_' + battery + '.sav'
    else:
        filename = 'Bayesian_Ridge_' + battery + '.sav'


    path_for_model = [] # save trained model
    pickle.dump(model, open(path_for_model + filename, 'wb'))
