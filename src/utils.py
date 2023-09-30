# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import requests
import sqlite3

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

import xgboost

import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout

from config import *

import warnings
warnings.filterwarnings("ignore")

# set numpy and tensorflow random seed
np.random.seed(seed = SEED_VALUE)
tf.random.set_seed(seed = SEED_VALUE)

def download_data(url, fn):
    '''
    Download file from a website and save it to the data path

    Parameters
    url: website url where the data resides
    fn: filename
    '''
    # Ensure the target folder exists, create it if not
    os.makedirs(PATH_DATA, exist_ok=True)

    # Send an HTTP GET request to the URL
    response = requests.get(url)

    if response.status_code == 200:
        # If the request is successful, save the content to a file
        file_path = os.path.join(PATH_DATA, fn)
    
        with open(file_path, 'wb') as file:
            file.write(response.content)
    
        print(f"File saved to: {file_path}")
        
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")


def load_data(db_file, db_name):
    '''
    Create a database connection to the SQLite database specified by the db_file and query data from data source connection and load it into a dataframe
    
    Parameters
    db_file: database file
    db_name: database name
    
    Return
    df: Data Frame object
    '''
    query = f" SELECT * FROM {db_name}"
    
    connection = sqlite3.connect(db_file)
    cursor = connection.cursor()
    cursor.execute(query)
    raw_data = cursor.fetchall()
    raw_data_column_name = list(map(lambda x: x[0], cursor.description))
    
    df = pd.DataFrame(raw_data)
    df.columns = raw_data_column_name

    return df


def preprocess_data(df):
    '''
    Parameters
    df: data source that will be preprocessed
    
    Return 
    df: data frame object
    '''
    # drop missing final test data
    df = df[~df['final_test'].isna()]
    
    # ensure that all students are at least 15 year old
    df = df[df['age']>=15]

    # ensure that all CCA are in uppercase to ensure consistent data entry
    df['CCA'] = df['CCA'].str.upper()

    # ensure all tuition are inputted as Yes or No
    df['tuition'] = np.where(df['tuition']=='Y', 'Yes', df['tuition'])
    df['tuition'] = np.where(df['tuition']=='N', 'No', df['tuition'])

    # derive sleep hours as it is more meaningful and drop wake_time and sleep_time 
    df['sleep_hours'] = np.where(pd.to_datetime(df['sleep_time']).dt.hour >= 12, 
                             (pd.to_datetime(df['wake_time']).dt.hour + 12) - (pd.to_datetime(df['sleep_time']).dt.hour - 12),
                             pd.to_datetime(df['wake_time']).dt.hour - pd.to_datetime(df['sleep_time']).dt.hour
                            )

    df.drop(['wake_time', 'sleep_time'], axis=1, inplace=True)

    # drop index, bag_color and student_id as they are not meaningful
    df.drop(['index', 'bag_color', 'student_id'], axis=1, inplace=True)

    # drop attendance_rateto avoid multi-colinearity with sleep_hours
    df.drop(['attendance_rate'], axis=1, inplace=True)
    
    return df


def linear_elasticnet_regressor_model(X_train, y_train):
    '''
    Parameters
    X_train = training independent variables
    y_train = training target variable
    
    Return 
    linreg: Linear Regression Elastic Net ML Model object
    '''
    linreg = ElasticNet()

    # use randomized search cv to look for the best hyperparameters combination
    linreg_random_search = RandomizedSearchCV(linreg, param_distributions=PARAM_GRID_LINREG_ELASTICNET, cv=K_FOLD_CV, scoring=SCORE_CV)
    linreg_random_search.fit(X_train, y_train)

    # fit the ML model with the best hyperparameters combination
    linreg = ElasticNet(alpha = linreg_random_search.best_params_['alpha'],
                        l1_ratio = linreg_random_search.best_params_['l1_ratio']
                       )

    linreg.fit(X_train, y_train)

    return linreg


def linear_ridge_regressor_model(X_train, y_train):
    '''
    Parameters
    X_train = training independent variables
    y_train = training target variable
    
    Return 
    linreg: Linear Regression Ridge ML Model object
    '''
    linreg = Ridge()

    # use randomized search cv to look for the best hyperparameters combination
    linreg_random_search = RandomizedSearchCV(linreg, param_distributions=PARAM_GRID_LINREG_RIDGE, cv=K_FOLD_CV, scoring=SCORE_CV)
    linreg_random_search.fit(X_train, y_train)

    # fit the ML model with the best hyperparameters combination
    linreg = Ridge(alpha = linreg_random_search.best_params_['alpha'],
                  )

    linreg.fit(X_train, y_train)

    return linreg


def linear_lasso_regressor_model(X_train, y_train):
    '''
    Parameters
    X_train = training independent variables
    y_train = training target variable
    
    Return 
    linreg: Linear Regression Lasso ML Model object
    '''
    linreg = Lasso()

    # use randomized search cv to look for the best hyperparameters combination
    linreg_random_search = RandomizedSearchCV(linreg, param_distributions=PARAM_GRID_LINREG_LASSO, cv=K_FOLD_CV, scoring=SCORE_CV)
    linreg_random_search.fit(X_train, y_train)

    # fit the ML model with the best hyperparameters combination
    linreg = Lasso(alpha = linreg_random_search.best_params_['alpha'],
                  )

    linreg.fit(X_train, y_train)

    return linreg


def xtreme_gradient_boosting_regressor_model(X_train, y_train):
    '''
    Parameters
    X_train = training independent variables
    y_train = training target variable

    Return 
    xgb: Xtreme Gradient Boosting Regressor ML Model object
    '''
    xgb = xgboost.XGBRegressor(random_state=SEED_VALUE)
    xgb_random_search = RandomizedSearchCV(xgb, param_distributions=PARAM_GRID_XGB, cv=K_FOLD_CV, scoring=SCORE_CV)
    xgb_random_search.fit(X_train, y_train)
    
    xgb = xgboost.XGBRegressor(max_depth=xgb_random_search.best_params_['max_depth'], 
                               learning_rate=xgb_random_search.best_params_['learning_rate'],
                               n_estimators=xgb_random_search.best_params_['n_estimators'],
                               colsample_bytree=xgb_random_search.best_params_['colsample_bytree'],
                               random_state=SEED_VALUE)
    xgb.fit(X_train, y_train)

    return xgb


def neural_network_regressor_model(X_train, y_train):
    '''
    Parameters
    X_train = training independent variables
    y_train = training target variable
    
    Return 
    nn: Neural Network Regressor ML Model object
    '''  
    nn = Sequential()
    nn.add(Dense(1, input_dim=X_train.shape[1]))
    nn.compile(loss = tf.keras.losses.mse,
               optimizer=tf.keras.optimizers.RMSprop(),
               metrics = ['mse']
              )
    
    history = nn.fit(X_train, y_train, validation_split=TRAIN_TEST_SPLIT_RATIO, epochs=N_EPOCHS, batch_size=BATCH_SIZE)

    # find the epochs with the minimum validation loss to avoid overfitting the model
    min_epoch = history.history['val_loss'].index(min(history.history['val_loss'])) + 1
    nn.fit(X_train, y_train, validation_split=TRAIN_TEST_SPLIT_RATIO, epochs=min_epoch, batch_size=BATCH_SIZE)
    
    return nn
    
    
def model_evaluation(model_list, X_test, y_test):
    '''
    Parameters
    model_list = list of model objects to compare its performance
    X_test = test independent data
    y_test = test target data
    
    Return 
    best_score: best score achieved by the best model object
    best_model: name of the best model performance
    '''
    prediction_model_table = {}
    
    best_score = float('inf')
    best_model = ''
    
    for model in model_list:
        y_pred = model_list[model].predict(X_test)

        prediction_model_table[model] = mean_squared_error(y_test, y_pred, squared=False)

        if mean_squared_error(y_test, y_pred) < best_score:
            best_score = mean_squared_error(y_test, y_pred, squared=False)
            best_model = model

    print(pd.DataFrame.from_dict(data=prediction_model_table, orient='index', columns=['RMSE']))
    print(f'Best ML model is {best_model} and achieved RMSE of {best_score}')

    return best_model


def model_explanability_xgb(xgb, X_test):
    '''
    Gain insight into the features that affecting the model explainability
    
    Parameters
    xgb = Xtreme Gradient Boosting ML Model object
    X_test = test independent data
    '''
    feature_names = X_test.columns
    xgb_importances = pd.Series(xgb.feature_importances_, index=feature_names).sort_values()
    
    y_pos = np.arange(len(X_test.columns))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(y_pos, xgb_importances)
    ax.set_yticks(y_pos, feature_names)
    
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig('XGBoost Feature Importance')

