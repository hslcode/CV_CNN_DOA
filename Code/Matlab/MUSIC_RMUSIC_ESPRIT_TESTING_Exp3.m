% CV-CNN Testing - Experiment 2:MUSIC, RMUSIC, ESPRIT, and CRLB.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Shulin Hu
% Date: 15/05/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
SNR_dB = 0;
T_vec = [20 50 100 200 300 400 500 600 700 800,900,1000];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Path of the test data
filename = fullfile('../../Data/EX3',...
    'EX3_Data_1Ktest_16ULA_K2_min10p3_min7p6_0dB_Snapshots_20to1000.h5');

% Location to save the result data
save_path = '../../Result/data/EX3';

% load the test data
r_sam = h5read(filename, '/SCM');
R_sam = squeeze(r_sam(:,:,1,:,:)+1j*r_sam(:,:,2,:,:));
True_angles = h5read(filename, '/angles');
SOURCE_K = size(True_angles,2);
[ULA_N,~, N_test,T_vec_size] = size(R_sam);
SOURCE.interval = 60;
SOURCE_power = ones(1, SOURCE_K);
res = 1;
noise_power = 10^(-SNR_dB/10);

% ESPIRT pars
ds = 1; % if the angle search space is lower than [-30,30] ds>1 can be used, e.g., ds=2--> u=1/ds=0.5 --> [-30,30] degrees 
ms = 8; % if 1 the weights are equal if ms>1 there are higher weights at the center elements of each subarray
w = min(ms,ULA_N-ds-ms+1);  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
% For the CRLB
der_a = @(x,N) 1j*(pi^2/180)*cos(deg2rad(x))*ULA_steer_vec(x,N).*(0:1:N-1)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
THETA_angles = -SOURCE.interval:res:SOURCE.interval;

% Initialization
RMSE_sam = zeros(1,T_vec_size);
RMSE_sam_rm = zeros(1,T_vec_size);
RMSE_sam_esp = zeros(1,T_vec_size);
CRB_uncr = zeros(1, T_vec_size);

for s=1:T_vec_size
    rmse_sam = 0;
    rmse_sam_rm = 0;
    rmse_sam_esp = 0;
    CRLBnit_uncr = 0;
    T = T_vec(s);
    for nit=1:N_test
        theta = True_angles';       
        A_ula = zeros(ULA_N,SOURCE_K);   
        B = zeros(ULA_N^2,SOURCE_K);
        D = zeros(ULA_N,SOURCE_K);
        D_uncr = zeros(ULA_N^2,SOURCE_K);
       for k=1:SOURCE_K  
        A_ula(:,k) = ULA_steer_vec(theta(k),ULA_N);   
        B(:,k) = kron(conj(A_ula(:,k)), A_ula(:,k));
        D(:,k) = der_a(theta(k),ULA_N);
        D_uncr(:,k) = kron(conj(D(:,k)), A_ula(:,k)) + kron(conj(A_ula(:,k)), D(:,k));
       end

    % These are used in the CRB for uncorrelated sources
    R = A_ula*A_ula' + noise_power*eye(ULA_N);
    R_inv = inv(R);
    PI_A = A_ula*pinv(A_ula);
    G = null(B');
    Rx_sam = R_sam(:,:,nit,s);

   % MUSIC estimator 
   [doas_sam, spec_sam, specang_sam] = musicdoa(Rx_sam,SOURCE_K, 'ScanAngles', THETA_angles);
   ang_sam  = sort(doas_sam)';
   ang_gt = sort(theta);
   % RMSE calculation
   rmse_sam = rmse_sam + norm(ang_sam- ang_gt)^2;
      
   % Root-MUSIC estimator 
   doas_sam_rm = sort(rootmusicdoa(Rx_sam, SOURCE_K))';
   ang_sam_rm = sort(doas_sam_rm);
   % RMSE calculation - degrees
   rmse_sam_rm = rmse_sam_rm + norm(ang_sam_rm - ang_gt)^2;
   
   % ESPRIT estimator
   doas_sam_esp = ESPRIT_doa(Rx_sam, ds, SOURCE_K, w);
   ang_sam_esp = sort(doas_sam_esp);
   rmse_sam_esp = rmse_sam_esp + norm(ang_sam_esp- ang_gt)^2;
   
   % CRB for uncorrelated sources
   C = kron(R.', R) + (noise_power^2/(ULA_N-SOURCE_K))*(PI_A(:)*(PI_A(:))');
   CRB_mat = inv(diag(SOURCE_power)*D_uncr'*G*inv(G'*C*G)*G'*D_uncr*diag(SOURCE_power))/T;
   CRLBnit_uncr = CRLBnit_uncr + real(trace(CRB_mat));
    
    end
    
    % MUSIC RMSE_deg
    RMSE_sam(s) = sqrt(rmse_sam/SOURCE_K/N_test);
    
    % R-MUSIC RMSE_deg
    RMSE_sam_rm(s) = sqrt(rmse_sam_rm/SOURCE_K/N_test);
    
    % ESPRIT RMSE_deg
    RMSE_sam_esp(s) = sqrt(rmse_sam_esp/SOURCE_K/N_test);
    
    CRB_uncr(s) = sqrt(CRLBnit_uncr/SOURCE_K/N_test);
    disp(['Processing number of Snapshot:', num2str(s)]);
end
%%
figure(1);
plot(T_vec,RMSE_sam,'^--');
hold on;
plot(T_vec,RMSE_sam_rm,'o--');
plot(T_vec,RMSE_sam_esp,'+--');
plot(T_vec, CRB_uncr,'.-');
hold off;
set(gca, 'YScale', 'log');
legend('MUSIC', 'R-MUSIC','ESPRIT','CRLB$_{uncr}$',...
    'interpreter','latex');
title('DoA-estimation of K=2 sources', 'interpreter','latex');
ylabel('RMSE [degrees]', 'interpreter','latex');
xlabel('T (snapshots)', 'interpreter','latex');
xticks([0.2 0.5 1 2 5 10 20 50 100]*100);
xticklabels([0.2 0.5 1 2 5 10 20 50 100]);
xlim([20 1000])
grid on;

% Save the results 
save(fullfile(save_path,'EX2_Result_MUSIC_RMUSIC_ESPRIT_CRB_1Ktest_16ULA_K2_min10p3_min7p6_0dB_Snapshots_20to1000.mat'),'T_vec','RMSE_sam','RMSE_sam_rm','RMSE_sam_esp','CRB_uncr');
