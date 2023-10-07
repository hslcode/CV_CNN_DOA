import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
from model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Select model
# model = torch.load(r'../../../Model/Pre_train/model_best.pth',map_location=device).to(device)
model = torch.load(r'../../../Model/Fine_tuning/model_best.pth',map_location=device).to(device)

# Test Data Path
f_data_root ='../../../Data/EX1/';f_result_root = '../../../Result/data/EX1/'
# Select one of the following four rows as the test data for this time
# f_data = f_data_root+'DOA_set_K_2_small.h5';f_result = f_result_root+'/EX1_result_DOA_set_K_2_small.h5'
# f_data = f_data_root+'DOA_set_K_2_lager.h5';f_result = f_result_root+'/EX1_result_DOA_set_K_2_lager.h5'
# f_data = f_data_root+'DOA_set_K_1.h5';f_result = f_result_root+'/EX1_result_DOA_set_K_1.h5'
f_data = f_data_root+'DOA_set_K_3.h5';f_result = f_result_root+'/EX1_result_DOA_set_K_3.h5'

# Load the test data
data_file = h5py.File(f_data, 'r')
GT_angles = np.transpose(np.array(data_file['angle'])) #Ground Truth angles
Ry_sam_test = np.array(data_file['SCM'])
test_data = Ry_sam_test
test_data = torch.tensor(test_data)
test_data = (test_data[0,:,:].type(torch.complex64)+1j*test_data[1,:,:].type(torch.complex64)).unsqueeze(0).to(device)

# Scenario prior parameters
soure_num = len(GT_angles[0])
res = 1
An_max = 60
An_min = -60
v = np.arange(An_min, An_max+res,res)

# DOA estimation obtained through model reconstruction of spatial spectrum
model.eval()
y_pred_sam = model(test_data).detach().cpu().numpy()
source_index = np.argpartition(y_pred_sam, -soure_num, axis=-1)[:, -soure_num:].reshape(-1)
DOA_esti = v[source_index]
print('Ground Truth angles:',GT_angles)
print('DOA Estimation results:',DOA_esti)

# Plot the reconstructed spatial spectrum
plt.figure()
plt_GT = plt.plot(GT_angles,1,"or",label = 'Ture DOA')
plt_Esti = plt.plot(DOA_esti,np.ones(len(DOA_esti)),"*b",label = 'Esti DOA')
plt.legend(handles=[plt_GT[0], plt_Esti[0]],labels=['Ture DOA','Esti DOA'])
plt.plot(v ,y_pred_sam[0,:])
plt.text(-60, 1.15, "GT:")
for i in GT_angles[0]:
        plt.text(i, 1.15, i)

plt.text(-60, 1.1, "esti:")
for i in DOA_esti:
        plt.text(i,1.1,i)
plt.xlabel('drictions(degree)')
plt.ylabel('probability')
plt.grid()
plt.show()

# Save the reconstructed spatial spectrum
if os.path.exists(f_result_root) is False:
        os.makedirs(f_result_root)
hf = h5py.File(f_result, 'w')
hf.create_dataset('GT_angles', data=GT_angles.T)
hf.create_dataset('spectrum', data=y_pred_sam)



