% TESTING DATA Generator - Experiment 2 (i.e., RMSE varies with SNR.)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Georgios K. Papageorgiou
%Modified by: Shulin Hu
% Date: 15/05/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
%clc;
tic;
rng(14);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save the data
filename = fullfile('../../Data/EX2',...
    'TEST_DATA1K_16ULA_K2_fixed_offgrid_ang_3D_min20to20SNR_T200_30_1_32_3.h5');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These parameters need to be set.
theta(1) = 30.1;        
theta(2) = 32.3; 
T = 200;                % number of snapshots
SNR_dB_vec = -20:5:20; % SNR values
SOURCE_K = 2;          % number of sources/targets - Kmax
Nsim = 1e+3;            %Number of Monte Carlo tests
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ULA_N = 16;
SOURCE.interval = 60;
res = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The steering/response vector of the ULA, where theta=0.5*sin(deg2rad(x));
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SOURCE_power = ones(1,SOURCE_K).^2;
THETA_angles = -SOURCE.interval:res:SOURCE.interval;
A_ula =zeros(ULA_N,SOURCE_K);
for k=1:SOURCE_K 
   A_ula(:,k) = ULA_steer_vec(theta(k),ULA_N);
end  

R_sam = zeros(ULA_N,ULA_N,3,Nsim,length(SNR_dB_vec));
R_the = zeros(ULA_N,ULA_N,3,Nsim,length(SNR_dB_vec));

parfor ii=1:length(SNR_dB_vec)
    SNR_dB = SNR_dB_vec(ii);
    noise_power = min(SOURCE_power)*10^(-SNR_dB/10);
    
    r_sam = zeros(ULA_N,ULA_N,3,Nsim);
    r_the = zeros(ULA_N,ULA_N,3,Nsim);
    
    for i=1:Nsim
    
    % The true covariance matrix 
    Ry_the = A_ula*diag(ones(SOURCE_K,1))*A_ula' + noise_power*eye(ULA_N);
    % The signal plus noise
    S = (randn(SOURCE_K,T)+1j*randn(SOURCE_K,T))/sqrt(2); 
    X = A_ula*S;
    Eta = sqrt(noise_power)*(randn(ULA_N,T)+1j*randn(ULA_N,T))/sqrt(2);
    Y = X + Eta;
    % The sample covariance matrix
    Ry_sam = Y*Y'/T;

    % Real and Imaginary part for the sample matrix 
    r_sam(:,:,1,i) = real(Ry_sam); 
    r_sam(:,:,2,i) = imag(Ry_sam);
    r_sam(:,:,3,i) = angle(Ry_sam);
    
    r_the(:,:,1,i) = real(Ry_the); 
    r_the(:,:,2,i) = imag(Ry_the);  
    r_the(:,:,3,i) = angle(Ry_the);
    end
    R_sam(:,:,:,:,ii) = r_sam;
    R_the(:,:,:,:,ii) = r_the;

ii
end
angles = theta;

time_tot = toc/60; % in minutes

h5create(filename,'/sam', size(R_sam));
h5write(filename, '/sam', R_sam);
h5create(filename,'/the', size(R_the));
h5write(filename, '/the', R_the);
h5create(filename,'/angles',size(angles));
h5write(filename, '/angles', angles);
