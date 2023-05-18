% DoA estimation via CNN: Training DATA generation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Georgios K. Papageorgiou
% Date: 18/9/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;
tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Location to save the DATA
filename = fullfile('E:\code\supp1-3089927\TSP_Deep_Networks_CODE_TSP\Code\Data\Train_Data_RQ_60',...
    'TRAIN_DATA_16ULA_K2_min20to10dB_res1_3D_60deg.h5');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SNR_dB_vec = -20:5:10; % SNR values
SOURCE_K = 2; % number of sources/targets - Kmax
ULA_N = 16;
SOURCE.interval = 60;
G_res = 1; % degrees
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The steering/response vector of the N-element ULA
ULA_steer_vec = @(x,N) exp(1j*pi*sin(deg2rad(x))*(0:1:N-1)).'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The training sets of angles 
ang_d0 = -SOURCE.interval:G_res:SOURCE.interval;
% ang_d1 = [ang_d0' NaN(length(ang_d0),1)];
ang_d2 = nchoosek(ang_d0,SOURCE_K);
ang_d = ang_d2;
S = length(SNR_dB_vec);
% r_sam = zeros(ULA_N, ULA_N,3,size(ang_d,1));
% R_sam = zeros([size(r_sam) S]);
L = size(ang_d,1);
r_the = zeros(ULA_N, ULA_N,3,L);
R_the = zeros([size(r_the) S]);
%*******************以下是为了生成采样协方差数据*****************/
r_sam = zeros(ULA_N, ULA_N,3,L);
R_sam = zeros([size(r_sam) S]);
%*******************end*****************/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Progress bar - comment while debugging
% pbar=waitbar(0,'Please wait...','Name','Progress');

parfor i=1:S
SNR_dB = SNR_dB_vec(i);
noise_power = 10^(-SNR_dB/10);
% Angle selection
% r_sam = zeros(ULA_N, ULA_N,3,size(ang_d,1));
r_the = zeros(ULA_N, ULA_N,3,L);
%*******************以下是为了生成采样协方差数据*****************/
r_sam = zeros(ULA_N, ULA_N,3,L);
%*******************end*****************/
for ii=1:L
    SOURCE_angles = ang_d(ii,:);
    A_ula = zeros(ULA_N,SOURCE_K);
    for k=1:SOURCE_K
        A_ula(:,k) = ULA_steer_vec(SOURCE_angles(k),ULA_N);
    end
% The true covariance matrix 
Ry_the = A_ula*diag(ones(SOURCE_K,1))*A_ula' + noise_power*eye(ULA_N);
%*******************以下是为了生成采样协方差数据*****************/
% The signal plus noise
T = 400;
S = (randn(SOURCE_K,T)+1j*randn(SOURCE_K,T))/sqrt(2); %注意包络也是随机信号
X = A_ula*S;
Eta = sqrt(noise_power)*(randn(ULA_N,T)+1j*randn(ULA_N,T))/sqrt(2);
Y = X + Eta;
Ry_sam = Y*Y'/T;
r_sam(:,:,1,ii) = real(Ry_sam); 
r_sam(:,:,2,ii) = imag(Ry_sam);
r_sam(:,:,3,ii) = angle(Ry_sam);
%*******************end*****************/
% Real and Imaginary part for the theor. covariance matrix R
r_the(:,:,1,ii) = real(Ry_the); 
r_the(:,:,2,ii) = imag(Ry_the);
r_the(:,:,3,ii) = angle(Ry_the);

end
i

R_the(:,:,:,:,i) = r_the;
%*******************以下是为了生成采样协方差数据*****************/
R_sam(:,:,:,:,i) = r_sam;
%*******************end*****************/
end

% The angles - Ground Truth
angles = ang_d;

% close(pbar);
time_tot = toc/60; % in minutes

% Save the DATA
h5create(filename,'/theor',size(R_the));
h5write(filename, '/theor', R_the);
h5create(filename,'/angles',size(angles));
h5write(filename, '/angles', angles);
% h5create(filename,'/sam',size(R_sam));
% h5write(filename, '/sam', R_sam);
h5disp(filename);
