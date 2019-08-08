function [y, klas, stats, Z] = sample_univ_TMoE(Alphak, Betak, Sigmak, Nuk, x)
% draw n samples from a univariate t mixture of experts (TMoE)
%
% Alphak: the parameters of the gating network
% Betak: the regression coefficients for the experts network
% Sigmak: the variances for the experts network
% Nuk: the degrees of freedom for the experts network t densities
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
    nuk = Nuk(Zik==1);
    %
%     Wi = gamrnd(nuk/2,nuk/2);
%     y(i) = normrnd(muk, sqrt(sigma2k/Wi));

     y(i) = muk + sigmak*trnd(nuk);

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
Vy_k = Nuk./(Nuk-2).*(Sigmak.^2);

% Var[yi]
Vy = sum(Piik.*(Ey_k.^2 + ones(n,1)*Vy_k),2) - Ey.^2;

stats.Ey_k = Ey_k;
stats.Ey = Ey;
stats.Vy_k = Vy_k;
stats.Vy = Vy;


