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
f_data_root ='../../../Data/EX2/'
f_data_root = f_data_root+'EX2_Data_1Ktest_16ULA_K2_T200_30p1_32p3_min20to20SNR.h5'

# The results save path
f_result_root = '../../../Result/data/EX2/'
f_result = f_result_root+'EX2_Result_CV_CNN_1Ktest_16ULA_K2_T200_30p1_32p3_min20to20SNR.h5'

# Load the test data
data_file = h5py.File(f_data_root, 'r')
GT_angles = np.transpose(np.array(data_file['angles'])) #Ground Truth angles
Ry_sam_test = torch.tensor(np.array(data_file['SCM']))
X_test_data_sam = (Ry_sam_test[:,:,0,:,:].type(torch.complex64)+1j*Ry_sam_test[:,:,1,:,:].type(torch.complex64)).unsqueeze(2).to(device)

# Scenario prior parameters
soure_num = len(GT_angles[0])
res = 1
An_max = 60
An_min = -60
v = np.arange(An_min, An_max+res,res)
print('Ground Truth angles:',GT_angles)

def RMSE_all_SNR(X_test,B, model,v):
    # The Monte Carlo experimental function
    model.eval()
    [SNRs, N_test,_,_,_] = X_test.shape
    if B.shape[0]==1:
        B = np.tile(B, reps=(N_test,1))
    RMSE = np.zeros((SNRs,1))
    for i in range(0,SNRs):
        X = X_test[i,:,:,:,:]
        y_pred_sam = model(X).detach().cpu().numpy()
        source_index = np.argpartition(y_pred_sam, -soure_num, axis=1)[:, -soure_num:]
        DOA_esti = np.sort(v[source_index])
        # Calculate the RMSE [in degrees]
        RMSE[i] = np.sqrt(mean_squared_error(np.sort(DOA_esti), np.sort(B)))
    return RMSE
# Perform Monte Carlo experiments and calculate RMSE
RMSE_CV_CNN_SNR = RMSE_all_SNR(X_test_data_sam,GT_angles, model,v)
print(RMSE_CV_CNN_SNR)

# Plot the RMSE
plt.figure()
plt.plot(RMSE_CV_CNN_SNR)
plt.yscale('log')
plt.xlabel('SNR(dB)')
plt.ylabel('RMSE(degree)')
plt.grid()
plt.show()

# Save the RMSE
if os.path.exists(f_result_root) is False:
        os.makedirs(f_result_root)
hf = h5py.File(f_result, 'w')
hf.create_dataset('CV_CNN_RMSE', data=RMSE_CV_CNN_SNR)
hf.close()