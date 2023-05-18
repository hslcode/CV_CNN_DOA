#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, ReLU, Softmax
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.initializers import glorot_normal
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import math
import tensorflow as tf
from tensorflow import keras


# In[2]:


# Model trained with batch size 32 and low SNR - 15/08/2020
model_CNN = load_model('../../../Model/Model_CNN_DoA_class_Data_N16_K2_res1_lowSNR_new_training_RQ_Adam_dropRoP_0_7.h5')
# Load the Test Data
filename_root ='../../../Data/EX3/'
filename2 = filename_root+'TEST_DATA1K_16ULA_K2_0dBSNR_3D_fixed_ang_vsT_min10_3_min7_6.h5'
#save path
f_result_root = '../../../Result/data/EX3/'
f_result = f_result_root+'RMSE_CNN_16ULA_K2_0dBSNR_fixed_ang_vsT_min10_3_min7_6.h5'

# In[3]:


res = 1
An_max = 60
An_min = -60
v = np.arange(An_min, An_max+res,res)


# In[4]:


f2 = h5py.File(filename2, 'r')
GT_angles = np.transpose(np.array(f2['angles']))

Ry_sam_test = np.array(f2['sam'])
X_test_data_sam = Ry_sam_test.swapaxes(2,4)



# In[5]:


B = GT_angles
print(B)


# In[6]:


def RMSE_all_SNR(X_test,B, model,v):
    [SNRs, N_test, N, M, Chan] = X_test.shape
    K = B.shape[1]
    if B.shape[0]==1:
        B = np.tile(B, reps=(N_test,1))
    RMSE = np.zeros((SNRs,1))
    for i in range(0,SNRs):
        X = X_test[i,:,:,:,:]
        with tf.device('/cpu:0'):
            x_pred = model.predict(X)
        x_ind_K = np.argpartition(x_pred, -K, axis=1)[:, -K:]
        A = np.sort(v[x_ind_K])
        # Calculate the RMSE [in degrees]
        RMSE[i] = np.sqrt(mean_squared_error(np.sort(A), np.sort(B)))
    return RMSE


# In[7]:


# RMSE_CNN_the = RMSE_all_SNR(X_test_data_the,B, model_CNN,v)
# print(RMSE_CNN_the)


# In[8]:


RMSE_CNN_sam = RMSE_all_SNR(X_test_data_sam,B, model_CNN,v)
print(RMSE_CNN_sam)


# In[9]:



hf = h5py.File(f_result, 'w')
hf.create_dataset('CNN_RMSE', data=RMSE_CNN_sam)
hf.close()

