# -*- coding: utf-8 -*-
"""
Created on Wed May 18 21:17:54 2022

@author: HP
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') # GPU or CPU

for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

#%% Paths
DATASET_TRAIN_PATH = os.path.join(os.getcwd(), 'Datasets', 'train.csv')
DATASET_TEST_PATH = os.path.join(os.getcwd(), 'Datasets', 'new_customers.csv')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model.h5')

#%% Loading of setting or models

model = load_model(PATH)
model.summary()

new_customers_segmentation = {A, B, C, D}


#%% Test Deployment 

customer_info = np.array([5,116,74,0,0,25.6,0.201,30])  # true label 0
customer_info_scaled = mms_scaler.transform(np.expand_dims(customer_info,
                                                           axis=0))

outcome = model.predict(customer_info_scaled)
print(np.argmax(outcome))
print(new_customer_segmentation[np.argmax(outcome)])


#%% Build your app 

with st.form('Customer segmentation form'):
    st.write("Customer's info")
    Gender = st.number_input('Gender')
    Ever_Married = st.number_input('Ever_Married')
    Age = int(st.number_input('Age'))
    Graduated = st.number_input('Graduated')
    Profesion = st.number_input('Profesion')
    Work_Experience = int(st.number_input('Work_Experience'))
    Spending_Score = st.number_input('Spending-Score')
    

    submitted = st.form_submit_button('submit')
    
    if submitted == True:
        customer_info = np.array([Gender,Ever_Married,Age,Graduated,
                                 Profesion,Work_Experience,Spending_Score])
        customer_info_scaled = mms.scaler.transform(np.expand_dims(customer_info,
                                                                  axis=0))
        
        outcome = model.predict(customer_info_scaled)
    
        st.write(new_customer_segmentation[np.argmax(outcome)])
    
      
            
            





























 






