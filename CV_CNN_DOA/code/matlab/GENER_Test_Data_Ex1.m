% DoA estimation via CV-DNN: Experiment 1 -spatial spectrum estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Shulin Hu
% Date: 5/15/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
tic;
rng(14);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Location to save the data
filename = fullfile('E:\code\last\Data\EX1','DOA_set_K_2_small.h5');
% Location for saving the l1-SVD results (no y data saved-just processed)
filename2 = fullfile('E:\code\last\Data\EX1','DOA_set_K_2_lager.h5');
% Location for saving the UnESPRIT results (no y data saved-just processed)
filename3 = fullfile('E:\code\last\Data\EX1','DOA_set_K_1.h5');
% Location for saving the UnESPRIT results (no y data saved-just processed)
filename4 = fullfile('E:\code\last\Data\EX1','DOA_set_K_3.h5');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 400; % number of snapreshots
SNR_dB = 0; % SNR values
DOA_set_K_2_small = [30.8 33.2];
DOA_set_K_2_lager = [-20.2 30];
DOA_set_K_1 = [-13.2];
DOA_set_K_3 = [-3 -30.8 10.2];
ULA.N = 16;
res = 1;%spatial resolution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[sam,angles] = Gener_Sam_Covar_Matrix(DOA_set_K_2_small,400,0,ULA.N);
h5create(filename,'/sam', size(sam));
h5write(filename, '/sam', sam);
h5create(filename,'/angle', size(angles));
h5write(filename, '/angle', angles);
[sam,angles] = Gener_Sam_Covar_Matrix(DOA_set_K_2_lager,400,0,ULA.N);
h5create(filename2,'/sam', size(sam));
h5write(filename2, '/sam', sam);
h5create(filename2,'/angle', size(angles));
h5write(filename2, '/angle', angles);
[sam,angles] = Gener_Sam_Covar_Matrix(DOA_set_K_1,400,0,ULA.N);
h5create(filename3,'/sam', size(sam));
h5write(filename3, '/sam', sam);
h5create(filename3,'/angle', size(angles));
h5write(filename3, '/angle', angles);
[sam,angles] = Gener_Sam_Covar_Matrix(DOA_set_K_3,400,0,ULA.N);
h5create(filename4,'/sam', size(sam));
h5write(filename4, '/sam', sam);
h5create(filename4,'/angle', size(angles));
h5write(filename4, '/angle', angles);
function [sam,angles] = Gener_Sam_Covar_Matrix(DOAs,T,SNR_dB,ULA_N)
    % The steering/response vector of the ULA, where theta=0.5*sin(deg2rad(x));
    ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
    K  = length(DOAs);
    A_ula =zeros(ULA_N,K);
    for k=1:K 
        A_ula(:,k) = ULA_steer_vec(DOAs(k),ULA_N);
    end  
    SOURCE.power = ones(1,K).^2;
    noise_power = min(SOURCE.power)*10^(-SNR_dB/10);
    S = (randn(K,T)+1j*randn(K,T))/sqrt(2); 
    X = A_ula*S;
    Eta = sqrt(noise_power)*(randn(ULA_N,T)+1j*randn(ULA_N,T))/sqrt(2);
    Y = X + Eta;
    Ry_sam = Y*Y'/T;
    sam(:,:,1) = real(Ry_sam); 
    sam(:,:,2) = imag(Ry_sam);
    angles(:) = DOAs';
end
