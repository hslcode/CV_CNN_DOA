import h5py
import matplotlib.pyplot as plt
from model import *
from data import MyDataset,My_Train_DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(r'../../../Model/Model_CV_CNN.pth',map_location='cpu').to(device)
f_data_root ='../../../Data/EX1/';f_result_root = '../../../Result/data/EX1/'
# f_data = f_data_root+'DOA_set_K_1.h5';f_result = f_result_root+'/EX1_result_DOA_set_K_1.h5'
# f_data = f_data_root+'DOA_set_K_2_lager.h5';f_result = f_result_root+'/EX1_result_DOA_set_K_2_lager.h5'
# f_data = f_data_root+'DOA_set_K_2_small.h5';f_result = f_result_root+'/EX1_result_DOA_set_K_2_small.h5'
f_data = f_data_root+'DOA_set_K_3.h5';f_result = f_result_root+'/EX1_result_DOA_set_K_3.h5'

data_file = h5py.File(f_data, 'r')
GT_angles = np.transpose(np.array(data_file['angle'])) #Ground Truth angles
Ry_sam_test = np.array(data_file['sam'])
test_data = Ry_sam_test
test_data = torch.tensor(test_data)
test_data = (test_data[0,:,:].type(torch.complex64)+1j*test_data[1,:,:].type(torch.complex64)).unsqueeze(0).to(device)

res = 1
An_max = 60
An_min = -60
v = np.arange(An_min, An_max+res,res)

model.eval()
y_pred_sam = model(test_data).detach().cpu().numpy()
plt.figure()
plt.plot(v ,y_pred_sam[0,:])
plt.plot(GT_angles,1,"or")
plt.grid()
plt.show()

hf = h5py.File(f_result, 'w')
hf.create_dataset('GT_angles', data=GT_angles.T)
hf.create_dataset('spectrum', data=y_pred_sam)



