% CV-CNN Testing - Experiment 2:MUSIC, RMUSIC, ESPRIT, and CRLB.
% Author: Georgios K. Papageorgiou
%Modified by: Shulin Hu
% Date: 15/05/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filename = fullfile('../../Data/EX2',...
    'TEST_DATA1K_16ULA_K2_fixed_offgrid_ang_3D_min20to20SNR_T200_30_1_32_3.h5');
save_path = '../../Result/data/EX2';

r_sam = h5read(filename, '/sam');
R_sam = squeeze(r_sam(:,:,1,:,:)+1j*r_sam(:,:,2,:,:));
r_the = h5read(filename, '/the');
R_the = squeeze(r_the(:,:,1,:,:)+1j*r_the(:,:,2,:,:));
True_angles = h5read(filename, '/angles');

SOURCE_K = size(True_angles,2);
[ULA_N,~, N_test,SNRs] = size(R_sam);
SOURCE_power = ones(1, SOURCE_K);
SOURCE.interval = 60;
res = 1;
SNR_vec = -20:5:20;

% UnESPRIT pars 
ds = 1; % if the angle search space is lower than [-30,30] ds>1 can be used, e.g., ds=2--> u=1/ds=0.5 --> [-30,30] degrees 
ms = 8; % if 1 the weights are equal if ms>1 there are higher weights at the center elements of each subarray
w = min(ms,ULA_N-ds-ms+1);  % Eq 9.133 in [1] 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For the CRLB
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
der_a = @(x,N) 1j*(pi^2/180)*cos(deg2rad(x))*ULA_steer_vec(x,N).*(0:1:N-1)';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A_ula = zeros(ULA_N,SOURCE_K);
D = zeros(ULA_N,SOURCE_K);
B = zeros(ULA_N^2,SOURCE_K);
D_uncr = zeros(ULA_N^2,SOURCE_K);
for k=1:SOURCE_K  
    A_ula(:,k) = ULA_steer_vec(True_angles(k),ULA_N);
    D(:,k) = der_a(True_angles(k),ULA_N);
    B(:,k) = kron(conj(A_ula(:,k)), A_ula(:,k));
    D_uncr(:,k) = kron(conj(D(:,k)), A_ula(:,k)) + kron(conj(A_ula(:,k)), D(:,k));
end
H = D'*(eye(ULA_N)-A_ula*pinv(A_ula))*D;
T = 400;

% These are used in the CRB for uncorrelated sources
PI_A = A_ula*pinv(A_ula);
G = null(B');

% Initialization
RMSE_the = zeros(1,SNRs);
RMSE_sam = zeros(1,SNRs);
RMSE_the_rm = zeros(1,SNRs);
RMSE_sam_rm = zeros(1,SNRs);
RMSE_the_esp = zeros(1,SNRs);
RMSE_sam_esp = zeros(1,SNRs);
CRLB = zeros(1,SNRs);
CRB_uncr = zeros(1,SNRs);

parfor s=1:SNRs

SNR_dB = SNR_vec(s);
noise_power = 10^(-SNR_dB/10);   
    
R = A_ula*A_ula' + noise_power*eye(ULA_N);
R_inv = inv(R);

rmse_the = 0;
rmse_sam = 0;
rmse_the_rm = 0;
rmse_sam_rm = 0;
rmse_the_esp = 0;
rmse_sam_esp = 0;


for nit=1:N_test
   
   % The true covariance matrix
   Rx = R_the(:,:,nit,s);
    
   % The smoothed sample covariance matrix
   Rx_sam = R_sam(:,:,nit,s);
    
   % MUSIC estimator 
   [doas_the, spec_the, specang_the] = musicdoa(Rx,SOURCE_K,'ScanAngles', -60:res:60);
   [doas_sam, spec_sam, specang_sam] = musicdoa(Rx_sam,SOURCE_K, 'ScanAngles', -60:res:60);
  
   ang_the = sort(doas_the)';
   ang_sam  = sort(doas_sam)';
   ang_gt = sort(True_angles)';
   
   % RMSE calculation
   rmse_the = rmse_the + norm(ang_the - ang_gt)^2;
   rmse_sam = rmse_sam + norm(ang_sam- ang_gt)^2;
      
   % Root-MUSIC estimator 
   doas_the_rm = sort(rootmusicdoa(Rx, SOURCE_K))';
   doas_sam_rm = sort(rootmusicdoa(Rx_sam, SOURCE_K))';

   ang_the_rm= sort(doas_the_rm);
   ang_sam_rm = sort(doas_sam_rm);
   
   % RMSE calculation - degrees
   rmse_the_rm = rmse_the_rm + norm(ang_the_rm - ang_gt)^2;
   rmse_sam_rm = rmse_sam_rm + norm(ang_sam_rm - ang_gt)^2;
   
    %% ESPRIT (with variable ds and reweighting technique)   
   % EPSRIT
   doas_the_esp = ESPRIT_doa(Rx, ds, SOURCE_K, w);
   doas_sam_esp = ESPRIT_doa(Rx_sam, ds, SOURCE_K, w);
     
   ang_the_esp = sort(doas_the_esp);
   ang_sam_esp = sort(doas_sam_esp);
   
   % ang = espritdoa(Rx_sam,SOURCE_K);
   rmse_the_esp = rmse_the_esp + norm(ang_the_esp - ang_gt)^2;
   rmse_sam_esp = rmse_sam_esp + norm( ang_sam_esp- ang_gt)^2;
   
end

% MUSIC RMSE_deg
RMSE_the(s) = sqrt(rmse_the/SOURCE_K/N_test);
RMSE_sam(s) = sqrt(rmse_sam/SOURCE_K/N_test);

% R-MUSIC RMSE_deg
RMSE_the_rm(s) = sqrt(rmse_the_rm/SOURCE_K/N_test);
RMSE_sam_rm(s) = sqrt(rmse_sam_rm/SOURCE_K/N_test);

% ESPRIT RMSE_deg
RMSE_the_esp(s) = sqrt(rmse_the_esp/SOURCE_K/N_test);
RMSE_sam_esp(s) = sqrt(rmse_sam_esp/SOURCE_K/N_test);

% Cramer-Rao lower bound
C_Cr = (noise_power/(2*T))*inv(real(H.*(A_ula'*R_inv*A_ula).'));
CRLB(s) = sqrt(trace(C_Cr)/SOURCE_K);

% CRB for uncorrelated sources
C = kron(R.', R) + (noise_power^2/(ULA_N-SOURCE_K))*(PI_A(:)*(PI_A(:))');
CRB_mat = inv(diag(SOURCE_power)*D_uncr'*G*inv(G'*C*G)*G'*D_uncr*diag(SOURCE_power))/T;
CRB_uncr(s) = sqrt(real(trace(CRB_mat))/SOURCE_K);
s
end

figure(1);
plot(SNR_vec,RMSE_sam,'s--');
hold on;
plot(SNR_vec,RMSE_sam_rm,'d--');
plot(SNR_vec,RMSE_sam_esp,'+--')
%plot(SNR_vec, CRLB,'*-');
plot(SNR_vec, CRB_uncr,'o-');
hold off;
set(gca, 'YScale', 'log');
legend('MUSIC', 'R-MUSIC','ESPRIT','CRLB$_{uncr}$',...
    'interpreter','latex');
title('DoA-estimation of K=2 sources', 'interpreter','latex');
ylabel('RMSE [degrees]', 'interpreter','latex');
xlabel('SNR [dB]', 'interpreter','latex');
grid on;

% % %% Save the results 

save(fullfile(save_path,'MUSIC_RMUSIC_ESPRIT_TESTING_EX2_min20to20SNR_T200_30_1_32_3.mat'),'SNR_vec','RMSE_sam','RMSE_sam_rm','RMSE_sam_esp','CRLB','CRB_uncr');

