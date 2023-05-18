% Plot the RMSE results for Experiment 2 with Delta \theta =3.19 vs SNR
% Author: Georgios K. Papageorgiou
% Date: 19/09/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear all;
close all;
clc;
close all;
% Load the results
T = 200; 
save_path = '../../Result/data/EX2';
% Load the MUSIC,R-MUSIC, CRLB results
File = fullfile(save_path,'MUSIC_RMUSIC_ESPRIT_TESTING_EX2_min20to20SNR_T200_30_1_32_3.mat');
load(File);

% Load the CNN results
filename = fullfile('../../Result/data/EX2',...
    'RMSE_CNN_16ULA_K2_min20to20SNR_T200_30_1_32_3.h5');   
RMSE_CNN = double(h5read(filename, '/CNN_RMSE'));
% Load the CV-CNN results  
filename = fullfile('../../Result/data/EX2',...
    'RMSE_CV_CNN_16ULA_K2_min20to20SNR_T200_3D_30_1_32_3.h5');   
RMSE_CV_CNN = double(h5read(filename, '/CV_CNN_RMSE'));

% New Color Options
orange = [0.8500, 0.3250, 0.0980];
gold_yellow = [0.9290, 0.6940, 0.1250];
new_green = [0.4660, 0.6740, 0.1880];

f=figure(1);
plot(SNR_vec,RMSE_sam,'^--','Color',orange,'LineWidth',1);
hold on;
plot(SNR_vec,RMSE_sam_rm,'o-.','Color',	gold_yellow,'LineWidth',1);
plot(SNR_vec,RMSE_sam_esp,'+--','Color','c','LineWidth',1);
plot(SNR_vec,RMSE_CNN,'x--','Color','b','LineWidth',1);
plot(SNR_vec,RMSE_CV_CNN,'d--','Color','r','LineWidth',1);
plot(SNR_vec, CRB_uncr,'.-','Color','k','LineWidth',1);
hold off;
set(gca, 'YScale', 'log');
legend('MUSIC', 'R-MUSIC','ESPRIT','CNN[11]','CV-CNN','CRLB$_{uncr}$',...
    'interpreter','latex');
% title(['RMSE of K=2([30.1,32.3]) sources from $T=$',num2str(T),' snapshots'], 'interpreter','latex');
ylabel('RMSE [$^\circ$]', 'interpreter','latex');
xlabel('SNR [dB]', 'interpreter','latex')
grid on;
