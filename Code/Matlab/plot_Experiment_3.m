% Plot the RMSE results for Exp. 3 RMSE vs Snapshots
% Author: Georgios K. Papageorgiou
% Modified by: Shulin Hu
% Date: 15/05/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
T_vec = [20 50 100 200 300 400 500 600 700 800,900,1000];
SNR = 0;
% Load the results
% Load the MUSIC,R-MUSIC, CRLB results
save_path = '../../Result/data/EX3';
File = fullfile(save_path,'EX2_Result_MUSIC_RMUSIC_ESPRIT_CRB_1Ktest_16ULA_K2_min10p3_min7p6_0dB_Snapshots_20to1000.mat');
load(File);
% Load the CNN results   
filename = fullfile('../../Result/data/EX3',...
        'EX3_Result_CNN_1Ktest_16ULA_K2_min10p3_min7p6_0dB_Snapshots_20to1000.h5');    
RMSE_CNN = double(h5read(filename, '/CNN_RMSE'));
% Load the CV-CNN results   
filename = fullfile('../../Result/data/EX3',...
        'EX3_Result_CV_CNN_1Ktest_16ULA_K2_min10p3_min7p6_0dB_Snapshots_20to1000.h5');    
RMSE_CV_CNN = double(h5read(filename, '/CV_CNN_RMSE'));

%% 
% New Color Options
orange = [0.8500, 0.3250, 0.0980];
gold_yellow = [0.9290, 0.6940, 0.1250];
new_green = [0.4660, 0.6740, 0.1880];

f=figure(1);
plot(T_vec,RMSE_sam,'^--','Color',orange,'LineWidth',1);
hold on;
plot(T_vec,RMSE_sam_rm,'o-.','Color',gold_yellow,'LineWidth',1);
plot(T_vec,RMSE_sam_esp,'+--','Color','g','LineWidth',1);

plot(T_vec,RMSE_CNN,'x--','Color','b','LineWidth',1);
plot(T_vec,RMSE_CV_CNN,'d--','Color','r','LineWidth',1);
plot(T_vec, CRB_uncr,'.-','Color','k','LineWidth',1);
hold off;
set(gca, 'YScale', 'log','XScale', 'log');
legend('MUSIC', 'R-MUSIC','ESPRIT','CNN in [12]','CV-CNN','CRLB$_{uncr}$',...
    'interpreter','latex');
title(['RMSE of K=2([-10.3,-7.6]) sources at ',num2str(SNR), ' dB SNR'], 'interpreter','latex');
ylabel('RMSE [$^\circ$]', 'interpreter','latex');
xlabel('T [snapshots] $\times 100$', 'interpreter','latex');
xticks([0.2 0.5 1 2 5 10 20 50 100]*100);
xticklabels([0.2 0.5 1 2 5 10 20 50 100]);
xlim([20 1000])
grid on;

% savefig(f,'RMSE_exp2_T100_cnn_res1_fixed_ang_offgrid.fig');