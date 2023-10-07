import sys
import os
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
from model import *
from regularizer import *
from data import MyDataset, My_Train_DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available(): torch.cuda.empty_cache()
# data file
filename_data_root = '../../../Data/Train/'
filename_data_Pre_train = filename_data_root + 'TRAIN_DATA_16ULA_K2_min20to10dB_res1_60deg_ECM.h5'
filename_data_Fine_tuning = filename_data_root + 'TRAIN_DATA_16ULA_K2_min20to10dB_res1_60deg_SCM.h5'
# file to save models
filename_save_root = '../../../Model/'
filename_save_model_Pre_train = filename_save_root+'Pre_train/'
filename_save_model_Fine_tuning =filename_save_root+ 'Fine_tuning/'
# file to save logs
filename_logs_root = '../../../Result/data/Train/'
writer_Pre_train = SummaryWriter(filename_logs_root+"Pre_train")
writer_Fine_tuning = SummaryWriter(filename_logs_root+"Fine_tuning")

# Pre_train parameters
epochs_Pre_train = 200
batch_size_Pre_train = 32
Cov_Matrix_type_Pre_train = 'ECM' # Pre-training stage: Use Expected Covariance Matrix
# Fine_tuning parameters
epochs_Fine_tuning = 100
batch_size_Fine_tuning = 64
Cov_Matrix_type_Fine_tuning = 'SCM' # Fine-tuning stage: Use Sampled Covariance Matrix
# dataset parameters
Validation_set_size = 0.2  # 80% for Training set and 20% for Validation set
#CV-CNN model
model = CV_CNN_Net(device).to(device)
def one_epoch(model,optimizer,lr,criterion,L1_Regularizer,dataloader,device,epoch_now,epoch_all,is_train,is_L1_Regularizer):
    if is_train==True:
        model.train()
    else:
        model.eval()
    mean_loss = torch.zeros(1).to(device)
    dataloader = tqdm(dataloader, desc=f'Epoch {epoch_now + 1}/{epoch_all}', file=sys.stdout)
    for step, data in enumerate(dataloader):
        x_batch, y_batch = data
        ouputs_train = model(x_batch.to(device))
        loss = criterion(ouputs_train, y_batch.to(device).to(torch.float32))
        if is_L1_Regularizer == True:
            for model_param_name, model_param_value in model.named_parameters():
                if model_param_name in ['FNN.0.weight']:
                    loss = L1_Regularizer.regularized_param(param_weights=model_param_value,
                                                                  reg_loss_function=loss)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        if is_train==True:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dataloader.desc = "[Training epoch {}] mean loss:{}".format(epoch_now, round(mean_loss.item(), 6))+' lr:{}'.format(lr)
        else:
            dataloader.desc = "[Val epoch {}] mean loss:{}".format(epoch_now,round(mean_loss.item(), 6)) + ' lr:{}'.format(lr)
    return mean_loss.item()
def Pre_Train():
    xTrain, xVal, yTrain, yVal = My_Train_DataLoader(filename_data_Pre_train, Validation_set_size,
                                                     Cov_Matrix_type_Pre_train)
    criterion = nn.BCELoss().to(device)
    L1_Regularizer = L1Regularizer(model, lambda_reg=5e-5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.85, 0.95), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=5, verbose=True)
    Train_dataloader = DataLoader(dataset=MyDataset(xTrain, yTrain), batch_size=batch_size_Pre_train, shuffle=True)
    Val_dataloader = DataLoader(dataset=MyDataset(xVal, yVal), batch_size=batch_size_Pre_train, shuffle=True)
    Val_loss_min = 99999.0
    for epoch in range(epochs_Pre_train):

        Train_loss = one_epoch(model,optimizer,optimizer.param_groups[0]['lr'],criterion,L1_Regularizer,
                               Train_dataloader,device,epoch,epochs_Pre_train,is_train=True,is_L1_Regularizer=True)


        Val_loss = one_epoch(model,optimizer,optimizer.param_groups[0]['lr'],criterion,L1_Regularizer,
                             Val_dataloader,device,epoch,epochs_Pre_train,is_train=False,is_L1_Regularizer=True)

        scheduler.step(Val_loss)

        writer_Pre_train.add_scalar("Train_loss", Train_loss, epoch)
        writer_Pre_train.add_scalar("Val_loss", Val_loss, epoch)
        writer_Pre_train.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) % 10 == 0:
            torch.save(model, filename_save_model_Pre_train + "/model_{}.pth".format(epoch + 1))
        if Val_loss_min > Val_loss:
            Val_loss_min = Val_loss
            torch.save(model, filename_save_model_Pre_train + "/model_best.pth")


def Fine_tuning():
    xTrain, xVal, yTrain, yVal = My_Train_DataLoader(filename_data_Fine_tuning, Validation_set_size,
                                             Cov_Matrix_type_Fine_tuning)
    # initialize Fine_tuning model with  paremeters of Pre_train model
    checkpoint = torch.load(filename_save_model_Pre_train + "/model_200.pth", map_location=device)
    model.load_state_dict(checkpoint.state_dict())

    criterion = nn.BCELoss().to(device)
    L1_Regularizer = L1Regularizer(model, lambda_reg=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.85, 0.95), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=5, verbose=True)
    Train_dataloader = DataLoader(dataset=MyDataset(xTrain, yTrain), batch_size=batch_size_Fine_tuning, shuffle=True)
    Val_dataloader = DataLoader(dataset=MyDataset(xVal, yVal), batch_size=batch_size_Fine_tuning, shuffle=True)

    Val_loss_min = 99999.0

    for epoch in range(epochs_Fine_tuning):

        Train_loss = one_epoch(model, optimizer, optimizer.param_groups[0]['lr'], criterion, L1_Regularizer,
                               Train_dataloader, device, epoch,epochs_Fine_tuning, is_train=True, is_L1_Regularizer=True)


        Val_loss = one_epoch(model, optimizer, optimizer.param_groups[0]['lr'], criterion, L1_Regularizer,
                             Val_dataloader, device,  epoch,epochs_Fine_tuning, is_train=False, is_L1_Regularizer=True)

        scheduler.step(Val_loss)

        writer_Fine_tuning.add_scalar("Train_loss", Train_loss, epoch)
        writer_Fine_tuning.add_scalar("Val_loss", Val_loss, epoch)
        writer_Fine_tuning.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) % 10 == 0:
            torch.save(model, filename_save_model_Fine_tuning + "/model_{}.pth".format(epoch + 1))
        if Val_loss_min > Val_loss:
            Val_loss_min = Val_loss
            torch.save(model, filename_save_model_Fine_tuning + "/model_best.pth")

if __name__ == "__main__":
    if os.path.exists(filename_save_model_Pre_train) is False:
        os.makedirs(filename_save_model_Pre_train)
    if os.path.exists(filename_save_model_Fine_tuning) is False:
        os.makedirs(filename_save_model_Fine_tuning)
    Pre_Train()
    Fine_tuning()

# ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9
# tensorboard --port 6007 --logdir




