'''

Select features

'''

import numpy as np
import pandas as pd
# import matplotlib2tikz
import matplotlib.pyplot as plt
import scipy.io
import pickle
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint, expon
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error, r2_score
# Scaling
from sklearn.metrics import mean_squared_error
# Confidence interval RF Specific
import forestci as fci
from matplotlib import gridspec



'''Random Forest'''


# import feature selection pipeline
folder = [] # path to pipeline_feature_selection_function
import sys
sys.path.append(folder)
from pipeline_feature_selection import pipeline_feature_selection

'''Do you want adversarial case?'''
adversarial = True

battery = i
data_fs = pd.read_csv(path + battery +'_fs.csv', index_col=0)
if adversarial:
    data_training = pd.read_csv(path + battery + '_adversarial_training.csv', index_col=0)
else:
    data_training = pd.read_csv(path + battery +'_training.csv', index_col=0)

data_test = pd.read_csv(path + battery + '_test.csv', index_col=0)
data_calibration = pd.read_csv(path + battery + '_calibration.csv', index_col=0)

# perform feature selection and transform the datasets
no_of_feature, rank_features, X_train, X_test, X_calibration, y_train, y_test, y_calibration, figure = pipeline_feature_selection(data_fs, data_training, data_test, data_calibration, 2)

print('Total number of features selected: {}'.format(no_of_feature))
print('\n Ranked features: {}'.format(rank_features))

# ''' save "regularised" datasets (features selected based on RF-RFE unsupervised) '''

data_test_fs = pd.concat([X_test, pd.DataFrame(y_test)], axis=1)
data_train_fs = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
data_calibration_fs = pd.concat([X_calibration, pd.DataFrame(y_calibration)], axis=1)

path_to_save = path_computer + 'Prognsotics/PhD Embedded Intelligence/Algorithm comparison on CALCE data/Data_for_comparison_paper/'

# save the data after feature selection for further used in the pipeline
if adversarial:
    data_train_fs.to_csv(path_to_save + 'data_train_fsed__adversarial' + battery + '.csv')
else:
    data_train_fs.to_csv(path_to_save + 'data_train_fsed_' + battery + '.csv')

data_test_fs.to_csv(path_to_save + 'data_test_fsed_' + battery + '.csv')
data_calibration_fs.to_csv(path_to_save + 'data_calibration_fsed_' + battery + '.csv')


