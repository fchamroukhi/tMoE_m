function [y, klas, stats, Z] = sample_univ_NMoLE(Alphak, Betak, Sigmak, x)%, n)
% draw n samples from a normal mixture of linear experts model
%
% Alphak: the parameters of the gating network
% Betak: the regression coefficients for the experts network
% Sigmak: the variances for the experts network
% x: the inputs (predictors)
%
% by Faicel Chamroukhi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


n = length(x);

p = size(Betak,1)-1;
q = size(Alphak,1)-1;
K = size(Betak,2);

% construct the regression design matrices
XBeta = designmatrix_Poly_Reg(x,p); % for the polynomial regression
XAlpha = designmatrix_Poly_Reg(x,q); % for the logistic regression


y = zeros(n,1);
Z = zeros(n,K);
klas = zeros(K,1);

%calculate the mixing proportions piik:
Piik = multinomial_logit(Alphak,XAlpha);
for i=1:n
    Zik = mnrnd(1,Piik(i,:));
    %
    muk = XBeta(i,:)*Betak(:,Zik==1);
    sigmak = Sigmak(Zik==1);
    %
    y(i) = normrnd(muk, sigmak);
    %
    Z(i,:) = Zik;
    zi = find(Zik==1);
    klas(i) = zi;
    %
end

% Statistics (means, variances)
% E[yi|zi=k]
Ey_k = XBeta*Betak;
% E[yi]
Ey = sum(Piik.*Ey_k,2);
% Var[yi|zi=k]
Vary_k = Sigmak.^2;
% Var[yi]
Vary = sum(Piik.*(Ey_k.^2 + ones(n,1)*Vary_k),2) - Ey.^2;

stats.Ey_k = Ey_k;
stats.Ey = Ey;
stats.Vary_k = Vary_k;
stats.Vary = Vary;


