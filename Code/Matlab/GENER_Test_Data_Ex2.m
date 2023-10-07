% TESTING DATA Generator - Experiment 2 (i.e., RMSE varies with SNR.)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Georgios K. Papageorgiou
% Modified by: Shulin Hu
% Date: 15/05/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
tic;
rng(14);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Location to save the data
filename = fullfile('../../Data/EX2',...ULA_M
    'EX2_Data_1Ktest_16ULA_K2_T200_30p1_32p3_min20to20SNR.h5');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% These parameters need to be set.
theta(1) = 30.1;            % The first Source
theta(2) = 32.3;            % The second Source
d = 0.5;                    % half-wavelength inter-element spacing 
T = 200;                    % number of snapshots
SNR_dB_vec = -20:5:20;      % SNR levels with a step of 5dB
SOURCE_K = 2;               % number of sources/targets
Nsim = 1e+3;                % Number of Monte Carlo tests
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ULA_M = 16;                 % element number of sensor array
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The steering/response vector of the ULA, where theta=0.5*sin(deg2rad(x));
ULA_steer_vec = @(x,N,d) exp(1j*2*pi*d*sind(x)*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SOURCE_power = ones(1,SOURCE_K).^2;
A_ula =zeros(ULA_M,SOURCE_K);
for k=1:SOURCE_K 
   A_ula(:,k) = ULA_steer_vec(theta(k),ULA_M,d);
end  

R_sam = zeros(ULA_M,ULA_M,3,Nsim,length(SNR_dB_vec));
parfor i=1:length(SNR_dB_vec)
    SNR_dB = SNR_dB_vec(i);
    noise_power = min(SOURCE_power)*10^(-SNR_dB/10);
    
    r_sam = zeros(ULA_M,ULA_M,3,Nsim);  % Temporary Sampled covariance variable at a SNR level
    
    for ii=1:Nsim
        S = (randn(SOURCE_K,T)+1j*randn(SOURCE_K,T))/sqrt(2); % Random source envelope signal
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

disp(['Processing SNR level:', num2str(i)]);
end

angles = theta;
h5create(filename,'/angles',size(angles));
h5write(filename, '/angles', angles);
h5create(filename,'/SCM', size(R_sam));
h5write(filename, '/SCM', R_sam);


