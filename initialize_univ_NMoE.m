function [Alphak, Betak, Sigma2k] = initialize_univ_NMoE(y, K, Xalpha, Xbeta, segmental)
%  initialize_univ_NMoE initializes the parameters of a univariate normal mixture-of-exerts
%
% Inputs:
%     y: n by 1 observed sampe
%     K: number of expert components
%     Xa: design mat of the softmax gating network
%     Xb: design mat of the Gaussian regressor experts network
%     seg: option, if the data are temporal (for segmentation) or not
% Outputs:
%     Alphak: matrix parameters of the softmax gating network
%     Betak: matrix parameters of the Gaussian regressor experts network
%     Sigma2k: vector of the variances of the regressors
%
% Faicel Chamroukhi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin==4, segmental = 0;end

n = length(y);
p = size(Xbeta,2) - 1;
q = size(Xalpha,2) - 1;


% Initialise the regression parameters (coeffecients and variances):
if ~segmental
    klas = randi(K,n,1);
    
    Betak = zeros(p+1,K);
    Sigma2k = zeros(1,K);
    for k=1:K
        Xk = Xbeta(klas==k,:);
        yk = y(klas==k);
        %the regression coefficients
        betak = Xk'*Xk\Xk'*yk;
        Betak(:,k) = betak;
        
        %the variances sigma2k
        Sigma2k(k)= sum((yk-Xk*betak).^2)/length(yk);
    end
else%segmental : segment uniformly the data and estimate the parameters
    nk = round(n/K)-1;
    Betak = zeros(p+1,K);
    Sigma2k = zeros(1,K);
    
    klas = zeros(n,1);
    for k=1:K
        yk = y((k-1)*nk+1:k*nk);
        Xk = Xbeta((k-1)*nk+1:k*nk,:);
        Betak(:,k) = Xk'*Xk\Xk'*yk;
        
        muk = Xk*Betak(:,k);
        sigma2k = (yk-muk)'*(yk-muk)/length(yk);%
        Sigma2k(k) =  sigma2k;
        
        %
        klas((k-1)*nk+1:k*nk)=k;
    end
end

% Intialise the softmax parameters
Alphak = zeros(q+1,K-1);
Z = zeros(n,K);
Z(klas*ones(1,K)==ones(n,1)*[1:K])=1;
Tau = Z;
res = IRLS(Xalpha, Tau, Alphak);
Alphak = res.W;