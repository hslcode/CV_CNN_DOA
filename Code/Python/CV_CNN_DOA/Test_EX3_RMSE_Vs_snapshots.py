import h5py
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torchstat import stat
from model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select model
# model = torch.load(r'../../../Model/Pre_train/model_best.pth',map_location=device).to(device)
model = torch.load(r'../../../Model/Fine_tuning/model_best.pth',map_location=device).to(device)

# Test Data Path
f_data_root ='../../../Data/EX3/'
f_data = f_data_root+'EX3_Data_1Ktest_16ULA_K2_min10p3_min7p6_0dB_Snapshots_20to1000.h5'

# The results save path
f_result_root = '../../../Result/data/EX3/'
f_result = f_result_root+'EX3_Result_CV_CNN_1Ktest_16ULA_K2_min10p3_min7p6_0dB_Snapshots_20to1000.h5'

# Load the test data
f2 = h5py.File(f_data, 'r')
GT_angles = np.transpose(np.array(f2['angles']))
Ry_sam_test = torch.tensor(np.array(f2['SCM']))
X_test_data_sam = (Ry_sam_test[:,:,0,:,:].type(torch.complex64)+1j*Ry_sam_test[:,:,1,:,:].type(torch.complex64)).unsqueeze(2).to(device)

# Scenario prior parameters
soure_num = len(GT_angles[0])
res = 1
An_max = 60
An_min = -60
v = np.arange(An_min, An_max+res,res)
print('Ground Truth angles:',GT_angles)

def RMSE_all_SNR(X_test,GT_angles, model,v):
    # The Monte Carlo experimental function
    model.eval()
    [SNRs, N_test,_,_,_] = X_test.shape
    if GT_angles.shape[0]==1:
        GT_angles = np.tile(GT_angles, reps=(N_test,1))
    RMSE = np.zeros((SNRs,1))
    for i in range(0,SNRs):
        X = X_test[i,:,:,:,:]
        x_pred = model(X).detach().cpu().numpy()
        x_ind_K = np.argpartition(x_pred, -soure_num, axis=1)[:, -soure_num:]
        A = np.sort(v[x_ind_K])
        # Calculate the RMSE [in degrees]
        RMSE[i] = np.sqrt(mean_squared_error(np.sort(A), np.sort(GT_angles)))
    return RMSE

# Perform Monte Carlo experiments and calculate RMSE
RMSE_CV_CNN_Snapshots = RMSE_all_SNR(X_test_data_sam,GT_angles, model,v)
print(RMSE_CV_CNN_Snapshots)

# Plot the RMSE
plt.figure()
plt.plot(RMSE_CV_CNN_Snapshots)
plt.xlabel('Snapshots')
plt.ylabel('RMSE(degree)')
plt.grid()
plt.show()

# Save the RMSE
if os.path.exists(f_result_root) is False:
        os.makedirs(f_result_root)
hf = h5py.File(f_result, 'w')
hf.create_dataset('CV_CNN_RMSE', data=RMSE_CV_CNN_Snapshots)
hf.close()