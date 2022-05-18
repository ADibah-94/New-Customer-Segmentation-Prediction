# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:43:31 2022

@author: HP
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import datetime 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#%% Paths

DATASET_TRAIN_PATH = os.path.join(os.getcwd(), 'Datasets', 'train.csv')
DATASET_TEST_PATH = os.path.join(os.getcwd(), 'Datasets', 'new_customers.csv')
LOG_PATH = os.path.join(os.getcwd(), 'logs') 
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')
                                  
#%% EDA

# Step 1) Data loading

train_df = pd.read_csv(DATASET_TRAIN_PATH)
test_df = pd.read_csv(DATASET_TEST_PATH)

# Step 2) Data Interpretation

train_df.head()         # to view the first rows of data
test_df.head()

train_df.describe()     # to view the statistics of data
test_df.describe()

train_df.info()         # to view the summary of dataframe
test_df.info()

# After check, there are columns in objects 
# so need to convert to numeric

train_df.isnull().sum()   # to check any missing values
test_df.isnull().sum()


# Step 3) Data cleaning

# Dealing with missing values 

train_df['Ever_Married'].fillna(method='ffill',inplace=True)
train_df['Graduated'].fillna(method='bfill',inplace=True)
train_df['Profession'].fillna(method='ffill',inplace=True)
train_df['Work_Experience'].fillna(method='bfill',inplace=True)
train_df['Family_Size'].fillna(method='bfill',inplace=True)

train_df.isna().sum()

test_df['Ever_Married'].fillna(method='ffill',inplace=True)
test_df['Graduated'].fillna(method='bfill',inplace=True)
test_df['Profession'].fillna(method='ffill',inplace=True)
test_df['Work_Experience'].fillna(method='bfill',inplace=True)
test_df['Family_Size'].fillna(method='bfill',inplace=True)

test_df.isna().sum()

# Change segmentation of train data to numeric

train = train_df[['Gender', 'Ever_Married', 'Age', 'Graduated', 
               'Profession','Work_Experience', 'Spending_Score', 
               'Family_Size', 'Segmentation']]

train.groupby('Gender').mean()
train.groupby('Ever_Married').mean()
train.groupby('Graduated').mean()
train.groupby('Profession').mean()
train.groupby('Spending_Score').mean()

train['Gender'] = train['Gender'].map({'Female':0,
                                      'Male':1})

train['Ever_Married'] = train['Ever_Married'].map({'No':0,
                                                 'Yes':1})

train['Graduated'] = train['Graduated'].map({'No':0,
                                           'Yes':1})

train['Profession'] = train['Profession'].map({'Artist':0, 
                                               'Doctor':1, 
                                               'Engineer':2,
                                               'Entertainment':3,
                                               'Executive':4,
                                               'Healthcare':5,
                                               'Homemaker':6,
                                               'Lawyer':7, 
                                               'Marketing':8})

train['Spending_Score'] = train['Spending_Score'].map({'Average':0,
                                                     'High':1,
                                                     'Low':2})

train['Segmentation'].value_counts().sort_values(ascending=False)

train.info()    # to view the summary of dataframe


# Change segmentation of test data to numeric

test = test_df[['Gender', 'Ever_Married', 'Age', 'Graduated', 
               'Profession','Work_Experience', 'Spending_Score', 
               'Family_Size', 'Segmentation']]

test.groupby('Gender').mean()
test.groupby('Ever_Married').mean()
test.groupby('Graduated').mean()
test.groupby('Profession').mean()
test.groupby('Spending_Score').mean()

test['Gender'] = test['Gender'].map({'Female':0,
                                     'Male':1})

test['Ever_Married'] = test['Ever_Married'].map({'No':0,
                                                 'Yes':1})

test['Graduated'] = test['Graduated'].map({'No':0,
                                           'Yes':1})

test['Profession'] = test['Profession'].map({'Artist':0, 
                                             'Doctor':1, 
                                              'Engineer':2,
                                              'Entertainment':3,
                                              'Executive':4,
                                              'Healthcare':5,
                                              'Homemaker':6,
                                              'Lawyer':7, 
                                              'Marketing':8})

test['Spending_Score'] = test['Spending_Score'].map({'Average':0,
                                                       'High':1,
                                                       'Low':2})

test['Segmentation'].value_counts().sort_values(ascending=False)

test.info()    # to view the summary of dataframe

# Combine train and test
data = pd.concat((train,test), ignore_index=True)
data.shape

data.info()
data.isnull().sum()

# Step 4) Features Selection

X = train.iloc[:,0:8]              # to get the features
Y = train.iloc[:,8]                # target

# Step 5) Data preprocessing

y = OneHotEncoder(sparse=False).fit_transform(np.expand_dims(Y, axis=-1))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

#%% Callbacks

log_dir = os.path.join(LOG_PATH, 
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

# Tensorboard callback

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Early stopping callback

early_stopping_callback= EarlyStopping(monitor='loss', patience=3)


 #%% Model Creation

model = Sequential()        # To create a container
model.add(Input(shape=(8)))    # Column of the Features
model.add(Dense(128, activation='sigmoid'))       # Hidden layer 1
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='sigmoid'))       # Hidden layer 2
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))            # output layer
model.summary()

#%% Compile and Model Fitting

model.compile(optimizer='adam',
              loss='mse',
              metrics='mse')

hist = model.fit(X_train,y_train, 
                 epochs=100,
                 validation_data=(X_test,y_test),
                 callbacks=[tensorboard_callback,early_stopping_callback])

#%% Model Evaluation and Analysis

pred_x = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(pred_x, axis=1)

print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))

#%% Model deployment

model.save(MODEL_SAVE_PATH)

#%% Discussion

# From the result, the accuracy achieved really low which is only 24%.
# It is because, the process of cleaning data does not clean properly.
# Besides, not enough data provided for training.
# Deep learning models usually require a lot of data for training.
# The more the data, the better will be the performance of the model.
# In future, to improve accuracy :
# 1) Get more data : 
#   The quality of models is generally constrained by the quality of training data.
# 2) Clean data properly
# 3) Transform data :
#    Must really get to know the data. 
#    Visualize it. 
#    Look for outliers.
# 3) Rescale data
# 4) Feature Selection :
#    There are lots of feature selection methods and feature importance methods 
#    Which can give ideas of features to keep and features to boot.



