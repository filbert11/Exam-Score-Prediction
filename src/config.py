SEED_VALUE = 2023

# URL of the website you want to fetch data from
URL_DATA = 'https://techassessment.blob.core.windows.net/aiap-preparatory-bootcamp/score.db'

# the data parameters: data path, filename and data source name
PATH_DATA = 'data'
FN_DATA = 'score.db'
NAME_DB = 'score'

# list of categorical, ordinal, numerical and target variables
LIST_CAT_VAR = ['direct_admission', 'CCA', 'learning_style', 'tuition', 'gender', 'mode_of_transport']
LIST_NUM_VAR = ['number_of_siblings', 'n_male', 'n_female', 'age', 'hours_per_week', 'sleep_hours']
TARGET_VAR = 'final_test'

# Machine Learning Model Hyper-parameters
TRAIN_TEST_SPLIT_RATIO = 0.3
K_FOLD_CV = 10
SCORE_CV ='neg_mean_squared_error'

# Linear Regression Elastic Net
PARAM_GRID_LINREG_ELASTICNET = {'alpha' : [0.01, 0.1, 1, 10, 100],
                                'l1_ratio' : [0, 0.25, 0.5, 0.75, 1]
                               }

# Linear Regression Ridge
PARAM_GRID_LINREG_RIDGE = {'alpha' : [0.01, 0.1, 1, 10, 100]
                          }

# Linear Regression Lasso
PARAM_GRID_LINREG_LASSO = {'alpha' : [0.01, 0.1, 1, 10, 100]
                          }

# Xtreme Gradient Boosting Classifier
PARAM_GRID_XGB = {'max_depth': [3, 6, 10],
                  'learning_rate': [0.01, 0.05, 0.1],
                  'n_estimators': [100, 500, 1000],
                  'colsample_bytree': [0.3, 0.7]
                 }

# Neural Network
N_EPOCHS = 100
BATCH_SIZE = 16