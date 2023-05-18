import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torchstat import stat
from model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(r'../../../Model/Model_CV_CNN.pth',map_location='cpu').to(device)
# Load the Test Data
f_data_root ='../../../Data/EX2/'
f_data_root = f_data_root+'TEST_DATA1K_16ULA_K2_fixed_offgrid_ang_3D_min20to20SNR_T200_30_1_32_3.h5'
#save path
f_result_root = '../../../Result/data/EX2/'
f_result = f_result_root+'RMSE_CV_CNN_16ULA_K2_min20to20SNR_T200_3D_30_1_32_3.h5'

data_file = h5py.File(f_data_root, 'r')
GT_angles = np.transpose(np.array(data_file['angles'])) #Ground Truth angles
print(GT_angles)
Ry_sam_test = torch.tensor(np.array(data_file['sam']))
X_test_data_sam = (Ry_sam_test[:,:,0,:,:].type(torch.complex64)+1j*Ry_sam_test[:,:,1,:,:].type(torch.complex64)).unsqueeze(2).to(device)

K = 2
res = 1
An_max = 60
An_min = -60
v = np.arange(An_min, An_max+res,res)


def RMSE_all_SNR(X_test,B, model,v):
    model.eval()
    [SNRs, N_test, N, M, Chan] = X_test.shape
    K = B.shape[1]
    if B.shape[0]==1:
        B = np.tile(B, reps=(N_test,1))
    RMSE = np.zeros((SNRs,1))
    for i in range(0,SNRs):
        X = X_test[i,:,:,:,:]
        x_pred = model(X).detach().cpu().numpy()
        x_ind_K = np.argpartition(x_pred, -K, axis=1)[:, -K:]
        A = np.sort(v[x_ind_K])
        # Calculate the RMSE [in degrees]
        RMSE[i] = np.sqrt(mean_squared_error(np.sort(A), np.sort(B)))
    return RMSE

RMSE_CNN_sam = RMSE_all_SNR(X_test_data_sam,GT_angles, model,v)
print(RMSE_CNN_sam)
plt.figure()
plt.plot(RMSE_CNN_sam)
plt.show()

hf = h5py.File(f_result, 'w')
hf.create_dataset('CV_CNN_RMSE', data=RMSE_CNN_sam)
hf.close()