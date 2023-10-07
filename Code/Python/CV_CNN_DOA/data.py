import h5py
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    # need to overload
    def __len__(self):
        return len(self.label[:,0])

    # need to overload
    def __getitem__(self, idx):
        data = self.data[idx,:,:,:]
        label = self.label[idx,:]
        return data, label

def My_Train_DataLoader(filename,test_size,data = 'ECM'):
    f1 = h5py.File(filename, 'r')
    angles = np.transpose(np.array(f1['angles']))
    Ry_the = np.array(f1[data])
    [SNRs, n, chan, M, N] = Ry_the.shape
    X_data = Ry_the.reshape([SNRs * n, chan, N, M])
    mlb = MultiLabelBinarizer()
    yTrain_encoded = mlb.fit_transform(angles)#构造一个多标签分类矩阵
    Y_Labels = np.tile(yTrain_encoded, reps=(SNRs,1))#由于每个角度组合有SNRs种场景，因此将yTrain_encoded沿着第一个维度复制SNRs份构成整体训练集的label
    xTrain, xVal, yTrain, yVal = train_test_split(X_data, Y_Labels, test_size=test_size, random_state=42)  # checked
    xTrain = torch.tensor(xTrain)
    xTrain = (xTrain[:, 0, :, :].type(torch.complex64) + 1j * xTrain[:, 1, :, :].type(torch.complex64)).unsqueeze(1)
    xVal = torch.tensor(xVal)
    xVal = (xVal[:, 0, :, :].type(torch.complex64) + 1j * xVal[:, 1, :, :].type(torch.complex64)).unsqueeze(1)
    yTrain = torch.tensor(yTrain)
    yVal = torch.tensor(yVal)
    return xTrain, xVal, yTrain, yVal
