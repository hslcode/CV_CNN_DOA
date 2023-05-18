% TESTING DATA Generator for Experiment 3 (i.e., RMSE varies with Snapshots.)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Georgios K. Papageorgiou
%Modified by: Shulin Hu
% Date: 15/05/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
rng(2015);
tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save the data
filename = fullfile('../../Data/EX3',...
    'TEST_DATA1K_16ULA_K2_0dBSNR_3D_fixed_ang_vsT_min10_3_min7_6.h5');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SNR_dB = 0;
% These are the angles
theta(1) = -10.3;
theta(2) = -7.6;
T_vec = [20 50 100 200 300 400 500 600 700 800,900,1000]; 
SOURCE_K = 2; % number of sources/targets - Kmax
ULA_N = 16;
SOURCE.interval = 60;
Nsim = 1000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The steering/response vector of the ULA, where theta=0.5*sin(deg2rad(x));
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SOURCE.power = ones(1,SOURCE_K).^2;
noise_power = min(SOURCE.power)*10^(-SNR_dB/10);
% grid
res = 1;
THETA_angles = -SOURCE.interval:res:SOURCE.interval;
thresh_vec = [130 180 270 410 570 910 1280];


    
A_ula =zeros(ULA_N,SOURCE_K);
for k=1:SOURCE_K 
    A_ula(:,k) = ULA_steer_vec(theta(k),ULA_N);
end  

% Initialization
R_sam = zeros(ULA_N,ULA_N,3,Nsim,length(T_vec));

parfor ii=1:length(T_vec)
    T = T_vec(ii);  
    r_sam = zeros(ULA_N,ULA_N,3,Nsim);
    
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
    
    end
 R_sam(:,:,:,:,ii) = r_sam;   

ii
end

time_tot = toc/60; % in minutes


% Save the data
h5create(filename,'/sam', size(R_sam));
h5write(filename, '/sam', R_sam);
h5create(filename,'/angles',size(theta));
h5write(filename, '/angles', theta);

