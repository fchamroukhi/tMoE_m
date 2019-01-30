function [Alphak, Betak, Sigma2k] = initialize_univ_NMoE(y, K, Xalpha, Xbeta, segmental)
% init_modele_regression estime les parametres de regression initiaux d'un
% modele de regression à processus logistique cache,où la loi conditionnelle
% des observations est une gaussienne.
%
% Entrees :
%
%        y: signal
%        nsignal (notez que pour la partie parametrisation des signaux les
%        observations sont monodimentionnelles)
%        K : nbre d'états (classes) cachés
%        duree_signal : = duree du signal en secondes
%        fs : fréquence d'échantiloonnage des signaux en Hz
%        ordre_reg : ordre de regression olynomiale
%
% Sorties :
%
%
%         param : parametres initiaux du modele de
%         regression : structure contenant les champs :
%         1. betak : le vecteur parametre de regression associe a la classe k.
%         vecteur colonne de dim [(p+1)x1]
%         2. sigma2k(k) = variance de x(i) sachant z(i)=k; sigma2k(j) =
%         sigma^2_k.
%
%
%
% Faicel Chamroukhi, Novembre 2008
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin==4, segmental = 0;end

n = length(y);
p = size(Xbeta,2) - 1;
q = size(Xalpha,2) - 1;

% Intialise the softmax parameters
Alphak = rand(q+1,K-1);%initialisation aléatoire du vercteur param�tre du IRLS

% Initialise the regression parameters (coeffecients and variances):

if ~segmental
    Zik = zeros(n,K);
    klas = randi(K,n,1);
    Zik(klas*ones(1,K)==ones(n,1)*[1:K])=1;
    Tauik = Zik;
    
    Betak = zeros(p+1,K);
    Sigma2k = zeros(1,K);
    for k=1:K
        Xk = Xbeta.*(sqrt(Tauik(:,k)*ones(1,p+1)));
        yk = y.*sqrt(Tauik(:,k));
        
        %the regression coefficients
        betak = Xk'*Xk\Xk'*yk;
        Betak(:,k) = betak;
        
        %the variances sigma2k
        Sigma2k(k)= sum(Tauik(:,k).*((y-Xbeta*betak).^2))/sum(Tauik(:,k));
    end
else%segmental : segment uniformly the data and estimate the parameters
    nk = round(n/K)-1;
    Betak = zeros(p+1,K);
    Sigma2k = zeros(1,K);
    for k=1:K
        yk = y((k-1)*nk+1:k*nk);
        Xk = Xbeta((k-1)*nk+1:k*nk,:);
        Betak(:,k) = Xk'*Xk\Xk'*yk;
        
        muk = Xk*Betak(:,k);
        sigma2k = (yk-muk)'*(yk-muk)/length(yk);%
        Sigma2k(k) =  sigma2k;
    end
end

