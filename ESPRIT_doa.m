% ESPRIT algorithm for DoA estimation
% Author (implementation): Georgios K. Papageorgiou
% Date: 24/3/2021

function ang = ESPRIT_doa(R, ds, D, w)
% Row weighting
N = size(R,1);
Ns = N-ds; %number of elements in a subarray

% Row weighting
weights = diag(sqrt([1:w-1 w*ones(1,Ns-2*(w-1)) w-1:-1:1])); % Eq 9.132 in [1]
O = zeros(Ns,ds);

% Selection Matrices
Js1 = [weights O]; % Eq 9.134 in [1]
Js2 = [O weights]; % Eq 9.135 in [1]

% % Selection Matrices
% Js1 = [eye(Ns) zeros(Ns, ds)]; % Eq 9.134 in [1]
% Js2 = [zeros(Ns,ds) eye(Ns)]; % Eq 9.135 in [1]

% Check for positive semi definite
[eigenvects,sED] = eig((R+R')/2);  % ensure Hermitian
sED = diag(sED);
diagEigenVals = sED;

%Sort eigenvectors
[~,indx] = sort(diagEigenVals,'descend');
eigenvects = eigenvects(:,indx);

% Selecting subarray signal subspaces
Us1 = Js1*eigenvects(:,1:D);
Us2 = Js2*eigenvects(:,1:D);

% TLS-ESPRIT
C = [Us1';Us2']*[Us1 Us2];    % Eq. (9.123) in [1]
[U,~,~] = svd(C);             % C is 2*D x 2*D
V12 = U(1:D,D+1:2*D);         % D x D
V22 = U(D+1:2*D,D+1:2*D);     % D x D
psi = -V12/V22;               % Eq. (9.122) in [1]
psieig = eig(psi);
%   Extract angle information estimated from two subarrays based on the
%   distance of the phase center between subarrays.
doas = angle(psieig)/ds;

%Convert estimated angle in sin-space to degrees. This method is valid for
%ULA only.
elSpacing = 0.5; % half-sapced ULAs
u = doas/(2*pi*elSpacing);

% check whether all elements of u are within [-1,1]
idx = find(abs(u)<=1/ds);
if  length(idx) <D
    warning('Invalid Psi. Decrease ds...');
end
if isempty(idx)
    ang = zeros(1,0);
else
    ang = sort(asind(u(idx)));
end

end
    