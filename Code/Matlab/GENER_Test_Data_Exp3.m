% TESTING DATA Generator for Experiment 3 (i.e., RMSE varies with Snapshots.)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Georgios K. Papageorgiou
% Modified by: Shulin Hu
% Date: 15/05/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
rng(14);
tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Location to save the data
filename = fullfile('../../Data/EX3',...
    'EX3_Data_1Ktest_16ULA_K2_min10p3_min7p6_0dB_Snapshots_20to1000.h5');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These parameters need to be set.
SNR_dB = 0;                                                 % The SNR
theta(1) = -10.3;                                           % The first Source
theta(2) = -7.6;                                            % The second Source
T_vec = [20 50 100 200 300 400 500 600 700 800,900,1000];   % The set of number of Snapshots
SOURCE_K = 2;                                               % number of sources/targets
ULA_M = 16;                                                  % element number of sensor array
Nsim = 1000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The steering/response vector of the ULA;
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SOURCE.power = ones(1,SOURCE_K).^2;
noise_power = min(SOURCE.power)*10^(-SNR_dB/10);
A_ula =zeros(ULA_M,SOURCE_K);
for k=1:SOURCE_K 
    A_ula(:,k) = ULA_steer_vec(theta(k),ULA_M);
end  

% Initialization
R_sam = zeros(ULA_M,ULA_M,3,Nsim,length(T_vec));

parfor i=1:length(T_vec)
    T = T_vec(i);  
    r_sam = zeros(ULA_M,ULA_M,3,Nsim);  % Temporary Sampled covariance variable at a SNR level
    for ii=1:Nsim
        % The signal plus noise
        S = (randn(SOURCE_K,T)+1j*randn(SOURCE_K,T))/sqrt(2);   % Random source envelope signal
        X = A_ula*S;
        Eta = sqrt(noise_power)*(randn(ULA_M,T)+1j*randn(ULA_M,T))/sqrt(2);
        Y = X + Eta;
        % The sampled covariance matrix
        Ry_sam = Y*Y'/T;
    
        % Real and Imaginary part of the SCM, and the angle part for the CNN model
        r_sam(:,:,1,ii) = real(Ry_sam); 
        r_sam(:,:,2,ii) = imag(Ry_sam);
        r_sam(:,:,3,ii) = angle(Ry_sam);  
    
    end
    R_sam(:,:,:,:,i) = r_sam;   

disp(['Processing number of Snapshot:', num2str(i)]);
end
% Save the data
h5create(filename,'/angles',size(theta));
h5write(filename, '/angles', theta);
h5create(filename,'/SCM', size(R_sam));
h5write(filename, '/SCM', R_sam);


