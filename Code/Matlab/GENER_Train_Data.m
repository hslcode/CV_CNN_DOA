% Training DATA generation
%*************************************************************************%
clear all;
close all;
clc;
tic;
%*************************************************************************%
% Location to save the DATA
filename_ECM = fullfile('../../Data/Train/','TRAIN_DATA_16ULA_K2_min20to10dB_res1_60deg_ECM.h5');
filename_SCM = fullfile('../../Data/Train/','TRAIN_DATA_16ULA_K2_min20to10dB_res1_60deg_SCM.h5');
%*************************************************************************%
SNR_dB_vec = -20:5:10;  % SNR levels 
SOURCE_K = 2;           % number of sources
ULA_M = 16;             % element number of sensor array
Max_DOA = 60;           % FOV:[-Max_DOA,Max_DOA]
grid_res = 1;           % resolution of the spaced discrete grids
d = 0.5;                % half-wavelength inter-element spacing
%*************************************************************************%
% The steering/response vector of the N-element ULA
ULA_steer_vec = @(x,N,d) exp(1j*2*pi*d*sin(deg2rad(x))*(0:1:N-1)).'; 
%*************************************************************************%
% The training sets
grids = -Max_DOA:grid_res:Max_DOA;  % spaced discrete grids
ang_d = nchoosek(grids,SOURCE_K);   % all the angle-pairs in degree 
S = length(SNR_dB_vec);             % number of the SNR levels
G = size(ang_d,1);                  % number of training angle-pairs
R_ECM = zeros(ULA_M, ULA_M,2,G,S);     % Expected covariance matrix (ECM)
R_SCM = zeros(ULA_M, ULA_M,2,G,S);     % Sampled covariance matrix (SCM)
%*************************************************************************%
parfor i=1:S
    SNR_dB = SNR_dB_vec(i);
    noise_power = 10^(-SNR_dB/10);
    r_the = zeros(ULA_M, ULA_M,2,G);    % Temporary expected covariance variable at a SNR level
    r_sam = zeros(ULA_M, ULA_M,2,G);    % Temporary Sampled covariance variable at a SNR level
    for ii=1:G
        SOURCE_angles = ang_d(ii,:);
        A_ula = zeros(ULA_M,SOURCE_K);
        for k=1:SOURCE_K
            A_ula(:,k) = ULA_steer_vec(SOURCE_angles(k),ULA_M,d);
        end
        % The Expected covariance matrix for a angle-pair
        Ry_the = A_ula*diag(ones(SOURCE_K,1))*A_ula' + noise_power*eye(ULA_M);
         % The Sampled covariance matrix for a angle-pair
        T = 400;    % the number of snapshots
        S = (randn(SOURCE_K,T)+1j*randn(SOURCE_K,T))/sqrt(2); % Random source envelope signal
        X = A_ula*S;
        Eta = sqrt(noise_power)*(randn(ULA_M,T)+1j*randn(ULA_M,T))/sqrt(2); % the number of snapshots
        Y = X + Eta;
        Ry_sam = Y*Y'/T; 
    
        % Real and Imaginary part for the ECM
        r_the(:,:,1,ii) = real(Ry_the); 
        r_the(:,:,2,ii) = imag(Ry_the);
        % Real and Imaginary part for the SCM
        r_sam(:,:,1,ii) = real(Ry_sam); 
        r_sam(:,:,2,ii) = imag(Ry_sam);
    end

    disp(['Processing SNR level:', num2str(i)]);
    R_ECM(:,:,:,:,i) = r_the;
    R_SCM(:,:,:,:,i) = r_sam;
end

% The angles Ground Truth
angles = ang_d;
% Save the DATA
h5create(filename_ECM,'/angles',size(angles));
h5write(filename_ECM, '/angles', angles);
h5create(filename_ECM,'/ECM',size(R_ECM));
h5write(filename_ECM, '/ECM', R_ECM);
h5disp(filename_ECM);
h5create(filename_SCM,'/angles',size(angles));
h5write(filename_SCM, '/angles', angles);
h5create(filename_SCM,'/SCM',size(R_SCM));
h5write(filename_SCM, '/SCM', R_SCM);
h5disp(filename_SCM);
