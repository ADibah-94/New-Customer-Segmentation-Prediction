# -*- coding: utf-8 -*-
"""
Created on Wed May 18 21:16:48 2022

@author: HP
"""
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout



class ExploratoryDataAnalysis():
    
    def __init(self):
        pass
    
    def remove_tags(self,data):
        '''


        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        '''
        
    
    def lower_split(self,data):
        '''
        This function converts all letters into lowercase and split into list.
        Also filters numerical data
        
        Parameters
        ----------
        data : Array
            RAW TRAINING DATA CONTAINING STRINGS.

        Returns
        -------
        data : List
            CLEANED DATA WITH ALL LETTERS CONVERTED INTO LOWERCASE.

        '''
        
        train = train_df[['Gender', 'Ever_Married', 'Age', 'Graduated', 
                          'Profession','Work_Experience', 'Spending_Score', 
                          'Family_Size', 'Segmentation']]


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



class ModelCreation():
    
    def __init__(self):
        pass
    
    
    def layer(self,nodes=128, dropout=0.2):
        
        mmodel = Sequential()        # To create a container
        model.add(Input(shape=(8)))    # Column of the Features
        model.add(Dense(128, activation='sigmoid'))       # Hidden layer 1
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='sigmoid'))       # Hidden layer 2
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='relu'))            # output layer
        model.summary()

        return model
    

#%% 
if __name__ == '__main__':
    
    import os
    import pandas as pd

    DATASET_TRAIN_PATH = os.path.join(os.getcwd(), 'Datasets', 'train.csv')
    DATASET_TEST_PATH = os.path.join(os.getcwd(), 'Datasets', 'new_customers.csv')
    LOG_PATH = os.path.join(os.getcwd(), 'logs') 
    MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')

    train_df = pd.read_csv(DATASET_TRAIN_PATH)
    test_df = pd.read_csv(DATASET_TEST_PATH)

    #%%
    
    train = train_df[['Gender', 'Ever_Married', 'Age', 'Graduated', 
                          'Profession','Work_Experience', 'Spending_Score', 
                          'Family_Size', 'Segmentation']]
    
    test = test_df[['Gender', 'Ever_Married', 'Age', 'Graduated', 
               'Profession','Work_Experience', 'Spending_Score', 
               'Family_Size', 'Segmentation']]
    
    data = pd.concat((train,test), ignore_index=True)
    pred_x = model.predict(X_test)
    model = load_model(PATH)
    
 





