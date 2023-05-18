% Plot the RMSE results for Exp. 3 RMSE vs T
% Author: Georgios K. Papageorgiou
% Date: 19/09/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear all;

% Load the results
T_vec = [20 50 100 200 300 400 500 600 700 800,900,1000];
SNR = 0;
% Load the MUSIC,R-MUSIC, CRLB results
save_path = '../../Result/data/EX3';
File = fullfile(save_path,'MUSIC_RMUSIC_ESPRIT_TESTING_EX3_0dB_vsT_min10_3_min7_6.mat');
load(File);
% Load the CNN results   
filename = fullfile('../../Result/data/EX3',...
        'RMSE_CNN_16ULA_K2_0dBSNR_fixed_ang_vsT_min10_3_min7_6.h5');    
RMSE_CNN = double(h5read(filename, '/CNN_RMSE'));
% Load the CV-CNN results   
filename = fullfile('../../Result/data/EX3',...
        'RMSE_CV_CNN_16ULA_K2_0dBSNR_fixed_ang_vsT_min10_3_min7_6.h5');    
RMSE_CV_CNN = double(h5read(filename, '/CV_CNN_RMSE'));
% Load the l1-SVD results
% filename2 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
%     'RMSE_l1_SVD_DATA1K_16ULA_K2_min10dBSNR_3D_fixed_ang_sep3coma6_vsT.h5');
% RMSE_l1SVD = double(h5read(filename2, '/RMSE_l1_SVD'));
% UnESPRIT
% filename3 = fullfile('C:\Users\geo_p\OneDrive - Heriot-Watt University\DoA DATA\DoA_DATA_JOURNALS',...
%     'RMSE_UnESPRIT_DATA1K_16ULA_K2_min10dBSNR_3D_fixed_ang_sep3coma6_vsT.h5');
% RMSE_UnESPRIT = double(h5read(filename3, '/RMSE_UnESPRIT'));

% RMSE_MLP = [1.61, 1.48, 0.83, 0.57, 0.67, 0.18, 0.21];
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
% plot(T_vec,RMSE_UnESPRIT,'s-.','Color','g');
% plot(T_vec,RMSE_l1SVD,'*--','Color','m');
% plot(T_vec,RMSE_MLP,'x--','Color','c');
plot(T_vec,RMSE_CNN,'x--','Color','b','LineWidth',1);
plot(T_vec,RMSE_CV_CNN,'d--','Color','r','LineWidth',1);
% plot(T_vec, CRLB,'.-','Color','k');
plot(T_vec, CRB_uncr,'.-','Color','k','LineWidth',1);
hold off;
set(gca, 'YScale', 'log','XScale', 'log');
% legend('MUSIC', 'R-MUSIC','ESPRIT','UnESPRIT','$\ell_{2,1}$-SVD','CNN','CRLB$_{uncr}$',...
%     'interpreter','latex');
legend('MUSIC', 'R-MUSIC','ESPRIT','CNN[11]','CV_CNN','CRLB$_{uncr}$',...
    'interpreter','latex');
% title(['RMSE of K=2([-10.8,-7.1]) sources at ',num2str(SNR), ' dB SNR'], 'interpreter','latex');
ylabel('RMSE [$^\circ$]', 'interpreter','latex');
xlabel('T [snapshots] $\times 100$', 'interpreter','latex');
% xticks([20 50 100 200 300 400 500 600 700 800,900,1000]);
% xticklabels([20 50 100 200 300 400 500 600 700 800,900,1000]);
xticks([0.2 0.5 1 2 5 10 20 50 100]*100);
xticklabels([0.2 0.5 1 2 5 10 20 50 100]);
xlim([20 1000])
grid on;

% savefig(f,'RMSE_exp2_T100_cnn_res1_fixed_ang_offgrid.fig');