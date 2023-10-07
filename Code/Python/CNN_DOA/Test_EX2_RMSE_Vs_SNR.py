#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import os
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
import math
import tensorflow as tf
from tensorflow import keras


# In[2]:


# Model trained with batch size 32 and low SNR - 15/08/2020
model_CNN = load_model('../../../Model/Model_CNN_DoA_class_Data_N16_K2_res1_lowSNR_new_training_RQ_Adam_dropRoP_0_7.h5')

# The test Data Path
f_data_root ='../../../Data/EX2/'
f_data_root = f_data_root+'EX2_Data_1Ktest_16ULA_K2_T200_30p1_32p3_min20to20SNR.h5'

# The results save path
f_result_root = '../../../Result/data/EX2/'
f_result = f_result_root+'EX2_Result_CNN_1Ktest_16ULA_K2_T200_30p1_32p3_min20to20SNR.h5'



# Load the test data
data_file = h5py.File(f_data_root, 'r')
GT_angles = np.transpose(np.array(data_file['angles']))
Ry_sam_test = np.array(data_file['SCM'])
X_test_data_sam = Ry_sam_test.swapaxes(2,4)

# Scenario prior parameters
soure_num = len(GT_angles[0])
res = 1
An_max = 60
An_min = -60
v = np.arange(An_min, An_max+res,res)
print('Ground Truth angles:',GT_angles)
def RMSE_all_SNR(X_test,GT_angles, model,v):
    [SNRs, N_test, _, _, _] = X_test.shape
    if GT_angles.shape[0]==1:
        GT_angles = np.tile(GT_angles, reps=(N_test,1))
    RMSE = np.zeros((SNRs,1))
    for i in range(0,SNRs):
        X = X_test[i,:,:,:,:]
        with tf.device('/cpu:0'):
            x_pred = model.predict(X)
        x_ind_K = np.argpartition(x_pred, -soure_num, axis=1)[:, -soure_num:]
        A = np.sort(v[x_ind_K])
        # Calculate the RMSE [in degrees]
        RMSE[i] = np.sqrt(mean_squared_error(np.sort(A), np.sort(GT_angles)))
    return RMSE

# Perform Monte Carlo experiments and calculate RMSE
RMSE_CNN_sam = RMSE_all_SNR(X_test_data_sam,GT_angles, model_CNN,v)
print(RMSE_CNN_sam)

# Save the RMSE
if os.path.exists(f_result_root) is False:
        os.makedirs(f_result_root)
hf = h5py.File(f_result, 'w')
hf.create_dataset('CNN_RMSE', data=RMSE_CNN_sam)
hf.close()

