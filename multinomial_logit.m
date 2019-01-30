function [probs, loglik] = multinomial_logit(W, X, Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [probs, loglik] = logit_model(W, X, Y)
%
% calculates the pobabilities according to multinomial logistic model:
%
% probs(i,k) = p(zi=k;W)= \pi_{ik}(W)
%                                  exp(wk'vi)
%                        =  ----------------------------
%                          1 + sum_{l=1}^{K-1} exp(wl'vi)
% for i=1,...,n and k=1...K
%
% Inputs :
%
%         1. W : parametre du modele logistique ,Matrice de dimensions
%         [(q+1)x(K-1)]des vecteurs parametre wk. W = [w1 .. wk..w(K-1)]
%         avec les wk sont des vecteurs colonnes de dim [(q+1)x1], le dernier
%         est suppose nul (sum_{k=1}^K \pi_{ik} = 1 -> \pi{iK} =
%         1-sum_{l=1}^{K-1} \pi{il}. vi : vecteur colonne de dimension [(q+1)x1]
%         qui est la variable explicative (ici le temps): vi = [1;ti;ti^2;...;ti^q];
%         2. M : Matrice de dimensions [nx(q+1)] des variables explicatives.
%            M = transpose([v1... vi ....vn])
%              = [1 t1 t1^2 ... t1^q
%                 1 t2 t2^2 ... t2^q
%                       ..
%                 1 ti ti^2 ... ti^q
%                       ..
%                 1 tn tn^2 ... tn^q]
%           q : ordre de regression logistique
%           n : nombre d'observations
%        3. Y Matrice de la partition floue (les probs a posteriori tik)
%           tik = p(zi=k|xi;theta^m); Y de dimensions [nxK] avec K le nombre de classes
% Sorties :
%
%        1. probs : Matrice de dim [nxK] des probabilites p(zi=k;W) de la vaiable zi
%          (i=1,...,n)
%        2. loglik : logvraisemblance du parametre W du modele logistique
%           loglik = Q1(W) = E(l(W;Z)|X;theta^m) = E(p(Z;W)|X;theta^m)
%                  = logsum_{i=1}^{n} sum_{k=1}^{K} tik log p(zi=k;W)
%
% Cette fonction peut egalement ?tre utilis?e pour calculer seulement les
% probs de la fa?oc suivante : probs = modele_logit(W,X)
%
% Faicel Chamroukhi 31 Octobre 2008 (mise ? jour)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin > 2
    [n1,K] = size(Y);
    [n2,q] = size(X);% here q is q+1
    if n1==n2
        n=n1;
    else
        error (' M et Y doivent avoir le meme nombre de ligne');
    end
else
    [n,q]=size(X);
end

if nargin > 2
    if size(W,2)== (K-1) % W doesnt contain the null vector associated with the last class
        wK=zeros(q,1);
        W = [W wK];% add the null vector wK for the last component probability
    elseif size(W,2)~=K
        error('W must have the dimension [(q+1)x(K-1)] or [(q+1)xK]');
    end
else
    wK=zeros(q,1);
    W = [W wK];% % add the null vector wK for the last component probability
    [q, K]= size(W);
end

XW = X*W;
maxm = max(XW,[],2);
XW = XW - maxm*ones(1,K);%to avoid overfolow
expXW = exp(XW);
piik = expXW./(sum(expXW(:,1:K),2)*ones(1,K));
% piik = normalize(expXW,2);
if nargin>2    %calcul de la log-vraisemblance
    loglik = sum(sum((Y.*XW) - (Y.*log(sum(expXW,2)*ones(1,K))),2));
    if isnan(loglik)
        % to avoid numerical overflow since exp(XW=-746)=0 and exp(XW=710)=inf)
        XW=X*W;
        minm = -745.1;
        XW = max(XW, minm);
        maxm = 709.78;
        XW= min(XW,maxm);
        expXW = exp(XW);
        
        % "log-likelihood"
        loglik = sum(sum((Y.*XW) - (Y.*log(sum(expXW,2)*ones(1,K)+eps)),2));
    end
    if isnan(loglik)
        Y
        error('Problem loglik IRLS NaN (!!!');
    end
else
    loglik = [];
end
probs = piik;