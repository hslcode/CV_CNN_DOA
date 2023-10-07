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
filename = fullfile('../../Data/EX1','DOA_set_K_2_small.h5');
filename2 = fullfile('../../Data/EX1','DOA_set_K_2_lager.h5');
filename3 = fullfile('../../Data/EX1','DOA_set_K_1.h5');
filename4 = fullfile('../../Data/EX1','DOA_set_K_3.h5');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 400;                        % number of snapreshots
SNR_dB = 0;                     % SNR values (dB)
DOA_set_K_2_small = [30.8 33.2];% two-signals scenario with a small angle separation
DOA_set_K_2_lager = [-20.2 30]; % two-signals scenario with a large angle separation
DOA_set_K_1 = [-13.2];          % single-signal scenario
DOA_set_K_3 = [-30.8 -3 10.2];  % three-signal scenario
ULA_M = 16;                     % element number of sensor array
d = 0.5;                        % half-wavelength inter-element spacing 
res = 1;                        % spatial resolution, i.e, resolution of the spaced discrete grids
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[sam,angles] = Gener_Sam_Covar_Matrix(DOA_set_K_2_small,T,SNR_dB,ULA_M,d);
h5create(filename,'/SCM', size(sam));
h5write(filename, '/SCM', sam);
h5create(filename,'/angle', size(angles));
h5write(filename, '/angle', angles);

[sam,angles] = Gener_Sam_Covar_Matrix(DOA_set_K_2_lager,T,SNR_dB,ULA_M,d);
h5create(filename2,'/SCM', size(sam));
h5write(filename2, '/SCM', sam);
h5create(filename2,'/angle', size(angles));
h5write(filename2, '/angle', angles);

[sam,angles] = Gener_Sam_Covar_Matrix(DOA_set_K_1,T,SNR_dB,ULA_M,d);
h5create(filename3,'/SCM', size(sam));
h5write(filename3, '/SCM', sam);
h5create(filename3,'/angle', size(angles));
h5write(filename3, '/angle', angles);

[sam,angles] = Gener_Sam_Covar_Matrix(DOA_set_K_3,T,SNR_dB,ULA_M,d);
h5create(filename4,'/SCM', size(sam));
h5write(filename4, '/SCM', sam);
h5create(filename4,'/angle', size(angles));
h5write(filename4, '/angle', angles);

function [sam,angles] = Gener_Sam_Covar_Matrix(DOAs,T,SNR_dB,ULA_M,d)
    % The steering/response vector of the ULA;
    ULA_steer_vec = @(x,N,d) exp(1j*2*pi*d*sin(deg2rad(x))*(0:1:N-1)).'; 
    K  = length(DOAs);
    A_ula =zeros(ULA_M,K);
    for k=1:K 
        A_ula(:,k) = ULA_steer_vec(DOAs(k),ULA_M,d);
    end  
    SOURCE.power = ones(1,K).^2;
    noise_power = min(SOURCE.power)*10^(-SNR_dB/10);
    S = (randn(K,T)+1j*randn(K,T))/sqrt(2); 
    X = A_ula*S;
    Eta = sqrt(noise_power)*(randn(ULA_M,T)+1j*randn(ULA_M,T))/sqrt(2);
    Y = X + Eta;
    Ry_sam = Y*Y'/T;
    sam(:,:,1) = real(Ry_sam); 
    sam(:,:,2) = imag(Ry_sam);
    angles(:) = DOAs';
end
