function solution = learn_univ_TMoE_EM(Y, x, K, p, q, total_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function solution = learn_TMoE_EM(X, K, p, q, fs, total_EM_tries, init_EM_kmeans, verbose_EM, verbose_IRLS)
%
% Learn a Regression model with a Hidden Logistic Process (RHLP) for
% modeling n curves (n>=1). The learning is performed by EM.
%
% Inputs :
%
%          1. X : Table of n curves each curve is composed of m points : dim(X)=[n m]
%                * Each curve is observed during the interval [0,T]=[t_1,...,t_m]
%                * t{j}-t_{j-1} = dt (sampling period)
%
%          2. K : Number of polynomial regression components (regimes)
%          3. p : degree of the polynomials
%          4. q :  order of the logistic regression (choose 1 for
%          convex segmentation)
%          5. fs : frequecy of sampling for the curves
%          6. total_EM_tries :  (the solution providing the highest log-lik
%          is chosen
%          7. verbose_EM : set to 1 for printing the "log-lik"  values during
%          EM iterations (by default verbose_EM = 0)
%          8. verbose_IRLS : set to 1 for printing the values of the criterion
%             optimized by IRLS at each IRLS iteration. (IRLS is used at
%             each M step of EM). (By default: verbose_EM = 0)
%
% Outputs :
%
%          1. solution : structure containing mainly the following fields:
%                      1.1 param : the model parameters:(W,beta1,...,betaK,sigma1,...,sigmaK).
%                          param is a structure containing the following
%                          fields:
%                          1.1.1 wk = (w1,...,wK-1) parameters of the logistic process:
%                          matrix of dimension [(q+1)x(K-1)] with q the
%                          order of logistic regression.
%                          1.1.2 betak = (beta1,...,betaK) polynomial
%                          regression coefficient vectors: matrix of
%                          dimension [(p+1)xK] p being the polynomial
%                          degree.
%                          1.1.3 sigmak = (sigma1,...,sigmak) : the
%                          variances for the K regmies. vector of dimension [Kx1]
%
%          4. tjk : post prob (fuzzy segmentation matrix of dim [mxK])
%          5. Zjk : Hard segmentation matrix of dim [mxK] obtained by the
%          MAP rule :  z_{jk} = 1 if and only z_j = arg max_k tjk
%          (k1,...,K)
%          appartient ï¿½ la classe k et zero sinon.
%          6. klas : column vector of the labels issued from Zjk, its
%          elements are klas(j)= k (k=1,...,K.)
%          8. theta : parameter vector of the model: theta=(wk,betak,sigmak).
%              column vector of dim [nu x 1] with nu = nbr of free parametres
%
%          9. Ey: curve expectation : sum of the polynomial components betak ri weighted by
%             the logitic probabilities pijk: Ey(j) = sum_{k=1}^K pijk betak rj, j=1,...,m. Ey
%              is a column vector of dimension m
%          13. ml : log-lik at convergence of EM
%          14. stored_loglik : vector of stored valued of the log-lik at each EM
%          iteration
%
%          17. BIC : valeur du critre BIC.  BIC = ml - nu*log(nm)/2.
%          17. ICL : valeur du critre ICL.  BIC = cl - nu*log(nm)/2.
%          18. AIC : valeur du critere AIC. AIC = ml - nu.
%          20. nu : nbr of free model parametres

%          21. phi :  Regression (covariate, Vendermond, design) matrices
%          22. XAlpha : design matrix for the logistic regression: matrix of dim [mx(q+1)].
%          23. XBeta : design matrix for the polynomial regression: matrix of dim [mx(p+1)].
%
%          2. log_fxj : logarithme des probabilites des
%          observations : log_fxj = log(sum_{i=1}^n sum_{k=1}^K pijk
%          fk(xi)). vecteur colonne de dim n
%          3. fxj : probas des observations : sum_{j=1}^m sum_{k=1}^K pijk
%          fk(xj)). vecteur colonne de dim n



% Faicel CHAMROUKHI
% Mise ï¿½ jour (01 Novembre 2008)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

warning off


if nargin<10 verbose_IRLS = 0; end
if nargin<9  verbose_IRLS =0; verbose_EM = 0; end;
if nargin<8  verbose_IRLS =0; verbose_EM = 0;   threshold = 1e-6; end;
if nargin<7  verbose_IRLS =0; verbose_EM = 0;   threshold = 1e-6; max_iter_EM = 1000; end;
if nargin<6  verbose_IRLS =0; verbose_EM = 0;   threshold = 1e-6; max_iter_EM = 1000; total_EM_tries=1;end;

if size(Y,2)==1, Y=Y'; end % cas d'une courbe

[n, m] = size(Y); % n curves, each curve is composed of m observations

% construct the regression design matrices
XBeta = designmatrix_Poly_Reg(x,p); % for the polynomial regression
XAlpha = designmatrix_Poly_Reg(x,q); % for the logistic regression

XBeta  = repmat(XBeta,n,1);
XAlpha = repmat(XAlpha,n,1);


y = reshape(Y',[],1);

best_loglik = -inf;
stored_cputime = [];
EM_try = 1;
while EM_try <= total_EM_tries,
    if total_EM_tries>1, disp(sprintf('EM run n°  %d  ',EM_try)); end
    time = cputime;
    %% EM Initialisation
    
    %%1. Initialisation of Alphak's, Betak's and Sigmak's
    segmental = 1;
    [Alphak, Betak, Sigma2k] = initialize_univ_NMoE(y,K, XAlpha, XBeta, segmental);
    if EM_try ==1, Alphak = zeros(q+1,K-1);end % set the first initialization to the null vector
    
    %%1. Initialisation the dof Nuk's
    %for k=1:K
    Nuk = 50*rand(1,K);
    %end
    %
    iter = 0;
    converge = 0;
    prev_loglik=-inf;
    stored_loglik=[];
    Alphak = Alphak;%
    %% EM %%%%
    while ~converge && (iter< max_iter_EM)
        iter=iter+1;
        %% E-Step
        Piik = multinomial_logit(Alphak,XAlpha);
        
        piik_fik = zeros(m*n,K);
        
        Wik = zeros(m*n,K);
        for k = 1:K
            
            muk = XBeta*Betak(:,k);
            sigma2k = Sigma2k(k);
            sigmak = sqrt(sigma2k);
            dik = (y - muk)/sigmak;
            %Dik(:,k) = dik;
            
            nuk = Nuk(k);
            % E[Wi|yi,zik=1]
            wik = (nuk + 1)./(nuk + dik.^2);
            Wik(:,k) = wik;
            
            % piik*STE(.;muk;sigma2k;lambdak)
            %weighted t linear expert likelihood
            piik_fik(:,k) = Piik(:,k).*pdf('tlocationscale', y, muk, sigmak, nuk);
            
            % see also tpdf
            
        end
        
        log_piik_fik = log(piik_fik);
        log_sum_piik_fik = log(sum(piik_fik,2));
        
        Tauik = piik_fik./(sum(piik_fik,2)*ones(1,K));
        
        %% M-Step
        % updates of alphak's, betak's, sigma2k's and lambdak's
        % --------------------------------------------------%
        % update of the softmax parameters (Alphak)
        %%  IRLS for multinomial logistic regression
        res = IRLS(XAlpha, Tauik, Alphak,verbose_IRLS);
        Piik = res.piik;
        Alphak = res.W;
        %%
        for k=1:K,
            XBetak = XBeta.*(sqrt(Tauik(:,k).*Wik(:,k))*ones(1,p+1));
            yk = y.*sqrt(Tauik(:,k).*Wik(:,k));
            
            %update the regression coefficients
            betak = XBetak'*XBetak\XBetak'*yk;
            Betak(:,k) = betak;
            
            % % update the variances sigma2k
            Sigma2k(k)= sum(Tauik(:,k).*Wik(:,k).*((y-XBeta*betak).^2))/sum(Tauik(:,k));%or change the divisor to sum(Tauik(:,k).*Wik(:,k))
            
            %% if ECM (use an additional E-Step with the updatated betak and sigma2k
            dik = (y - XBeta*Betak(:,k))/sqrt(Sigma2k(k));
            nuk = Nuk(k);
            % E[Wi|yi,zik=1]
            Wik(:,k) = (nuk + 1)./(nuk + dik.^2);
            
            %% update the nuk (the robustness parameter)
            nu0 = Nuk(k);
            Nuk(k) = fzero(@(nu) - psi((nu)/2) + log((nu)/2) + 1 ...
                + (1/sum(Tauik(:,k)))*sum(Tauik(:,k).*(log(Wik(:,k)) - Wik(:,k)))...
                + psi((Nuk(k) + 1)/2) - log((Nuk(k) + 1)/2), nu0);
        end
        
        
        %% observed-data log-likelihood
        loglik = sum(log_sum_piik_fik) + res.reg_irls;% + regEM;
        
        if verbose_EM,fprintf(1, 'EM - TMoE  : Iteration : %d   Log-lik : %f \n ',  iter,loglik); end
        converge = abs((loglik-prev_loglik)/prev_loglik) <= threshold;
        prev_loglik = loglik;
        stored_loglik = [stored_loglik, loglik];
    end% end of an EM loop
    EM_try = EM_try +1;
    stored_cputime = [stored_cputime cputime-time];
    
    %%% results
    param.Alphak = Alphak;
    param.Betak = Betak;
    param.Sigmak = sqrt(Sigma2k);
    param.Nuk = Nuk;
    solution.param = param;
    Piik = Piik(1:m,:);
    Tauik = Tauik(1:m,:);
    solution.Piik = Piik;
    solution.Tauik = Tauik;
    solution.log_piik_fik = log_piik_fik;
    solution.ml = loglik;
    solution.stored_loglik = stored_loglik;
    %% parameter vector of the estimated SNMoE model
    Psi = [param.Alphak(:); param.Betak(:); param.Sigmak(:); param.Nuk(:)];
    %
    solution.Psi = Psi;
    
    %% classsification pour EM : MAP(piik) (cas particulier ici to ensure a convex segmentation of the curve(s).
    [klas, Zik] = MAP(solution.Tauik);%solution.param.Piik);
    solution.klas = klas;
    
    % Statistics (means, variances)
    
    % E[yi|zi=k]
    Ey_k = XBeta(1:m,:)*Betak;
    solution.Ey_k = Ey_k;
    % E[yi]
    Ey = sum(Piik.*Ey_k,2);
    solution.Ey = Ey;
    
    % Var[yi|zi=k]
    Vary_k = Nuk./(Nuk-2).*Sigma2k;
    solution.Vary_k = Vary_k;
    
    % Var[yi]
    Vary = sum(Piik.*(Ey_k.^2 + ones(m,1)*Vary_k),2) - Ey.^2;
    solution.Vary = Vary;
    
    
    %%% BIC AIC et ICL
    lenPsi = length(Psi);
    solution.lenPsi = lenPsi;
    
    solution.BIC = solution.ml - (lenPsi*log(n*m)/2);
    solution.AIC = solution.ml - lenPsi;
    %% CL(theta) : complete-data loglikelihood
    zik_log_piik_fk = (repmat(Zik,n,1)).*solution.log_piik_fik;
    sum_zik_log_fik = sum(zik_log_piik_fk,2);
    comp_loglik = sum(sum_zik_log_fik);
    solution.CL = comp_loglik;
    solution.ICL = solution.CL - (lenPsi*log(n*m)/2);
    solution.XBeta = XBeta(1:m,:);
    solution.XAlpha = XAlpha(1:m,:);
    
    %%
    
    if total_EM_tries>1
        fprintf(1,'ml = %f \n',solution.ml);
    end
    if loglik > best_loglik
        best_solution = solution;
        best_loglik = loglik;
    end
end%fin de la premiï¿½re boucle while
solution = best_solution;
%
if total_EM_tries>1;   fprintf(1,'best loglik:  %f\n',solution.ml); end

solution.cputime = mean(stored_cputime);
solution.stored_cputime = stored_cputime;


