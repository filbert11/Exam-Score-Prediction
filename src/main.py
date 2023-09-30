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

from utils import *
from config import *

import shap

# set pandas column display to max
pd.set_option('display.max_columns', None)

# set numpy and tensorflow random seed
np.random.seed(seed = SEED_VALUE)
tf.random.set_seed(seed = SEED_VALUE)

if __name__ == "__main__":
    # extract data from data source and load it into dataframe
    # download data
    download_data(URL_DATA, FN_DATA)
    
    # load data
    df = load_data(db_file=os.path.join(PATH_DATA, FN_DATA), db_name=NAME_DB)

    # pre-process data
    df = preprocess_data(df)
    
    df_cat = pd.get_dummies(df[LIST_CAT_VAR], drop_first=True)
    
    df1 = pd.concat([df.drop(LIST_CAT_VAR, axis=1), df_cat], axis=1)

    X = df1.drop(TARGET_VAR, axis=1)
    y = df1[TARGET_VAR]

    # train test split the data
    # scale the data for numerical variables to ensure all numerical features have the same scale or range
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=SEED_VALUE)
    
    scaler = StandardScaler()
    X_train_num = pd.DataFrame(scaler.fit_transform(X_train[LIST_NUM_VAR]), columns=LIST_NUM_VAR)
    X_test_num = pd.DataFrame(scaler.transform(X_test[LIST_NUM_VAR]), columns=LIST_NUM_VAR)
    
    X_train = pd.concat([X_train_num, X_train.drop(LIST_NUM_VAR, axis=1).reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test_num, X_test.drop(LIST_NUM_VAR, axis=1).reset_index(drop=True)], axis=1)

    # train first model - Linear Regression with regularisation
    linreg_elasticnet = linear_elasticnet_regressor_model(X_train=X_train, y_train=y_train)
    linreg_ridge = linear_ridge_regressor_model(X_train=X_train, y_train=y_train)
    linreg_lasso = linear_lasso_regressor_model(X_train=X_train, y_train=y_train)

    # train second model - XGBoost Classifier
    xgb = xtreme_gradient_boosting_regressor_model(X_train=X_train, y_train=y_train)

    # train third model - Neural Network Classifier
    nn = neural_network_regressor_model(X_train=X_train, y_train=y_train)

    # compare the results of the model
    model_list = {
                  'Linear Regressor (Elastic Net)' : linreg_elasticnet,
                  'Linear Regressor (Ridge)' : linreg_ridge,
                  'Linear Regressor (Lasso)' : linreg_lasso,
                  'XGBoost' : xgb,
                  'Neural Network' : nn
                 }
    
    best_model = model_evaluation(model_list, X_test=X_test, y_test=y_test)

    # model explainability
    model_explanability_xgb(xgb, X_test)
