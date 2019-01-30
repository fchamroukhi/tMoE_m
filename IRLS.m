function res = IRLS(X, Y, Winit, verbose)
% res = IRLS(X, Y, Winit, verbose) : an efficient Iteratively Reweighted Least-Squares (IRLS) algorithm for estimating
% the parameters of a multinomial logistic regression model given the
% "predictors" X and a partition (hard or smooth) Y into K>=2 groups
%
% Inputs :
%
%         X : desgin matrix for the logistic weights.  dim(X) = [nx(q+1)]
%                            X = [1 t1 t1^2 ... t1^q
%                                 1 t2 t2^2 ... t2^q
%                                      ..
%                                 1 ti ti^2 ... ti^q
%                                      ..
%                                 1 tn tn^2 ... tn^q]
%            q being the number of predictors
%         Y : matrix of a hard or fauzzy partition of the data (here for
%         the RHLP model, Y is the fuzzy partition represented by the
%         posterior probabilities (responsibilities) (tik) obtained at the E-Step).
%
%         Winit : initial parameter values W(0). dim(Winit) = [(q+1)x(K-1)]
%         verbose : 1 to print the loglikelihood values during the IRLS
%         iterations, 0 if not
%
% Outputs :
%
%          res : structure containing the fields:
%              W : the estimated parameter vector. matrix of dim [(q+1)x(K-1)]
%                  (the last vector being the null vector)
%              piik : the logistic probabilities (dim [n x K])
%              loglik : the value of the maximized objective
%              LL : stored values of the maximized objective during the
%              IRLS training
%
%        Probs(i,k) = Pro(zi=k;W)
%                    = \pi_{ik}(W)
%                           exp(wk'vi)
%                    =  ---------------------------
%                      1+sum_{l=1}^{K-1} exp(wl'vi)
%
%       with :
%            * Probs(i,k) is the prob of component k at time t_i :
%            i=1,...n et k=1,...,K.
%            * vi = [1,ti,ti^2,...,ti^q]^T;
%       The parameter vecrots are in the matrix W=[w1,...,wK] (with wK is the null vector);
%

%% References
% Please cite the following papers for this code:
%
%
% @INPROCEEDINGS{Chamroukhi-IJCNN-2009,
%   AUTHOR =       {Chamroukhi, F. and Sam\'e,  A. and Govaert, G. and Aknin, P.},
%   TITLE =        {A regression model with a hidden logistic process for feature extraction from time series},
%   BOOKTITLE =    {International Joint Conference on Neural Networks (IJCNN)},
%   YEAR =         {2009},
%   month = {June},
%   pages = {489--496},
%   Address = {Atlanta, GA},
%  url = {https://chamroukhi.users.lmno.cnrs.fr/papers/chamroukhi_ijcnn2009.pdf}
% }
%
% @article{chamroukhi_et_al_NN2009,
% 	Address = {Oxford, UK, UK},
% 	Author = {Chamroukhi, F. and Sam\'{e}, A. and Govaert, G. and Aknin, P.},
% 	Date-Added = {2014-10-22 20:08:41 +0000},
% 	Date-Modified = {2014-10-22 20:08:41 +0000},
% 	Journal = {Neural Networks},
% 	Number = {5-6},
% 	Pages = {593--602},
% 	Publisher = {Elsevier Science Ltd.},
% 	Title = {Time series modeling by a regression approach based on a latent process},
% 	Volume = {22},
% 	Year = {2009},
% 	url  = {https://chamroukhi.users.lmno.cnrs.fr/papers/Chamroukhi_Neural_Networks_2009.pdf}
% 	}
%
% Faicel 31 octobre 2008 (mise ??? jour)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n, K] = size(Y);
[n,q] = size(X);% q here is (q+1)

if nargin<4; verbose = 0;end
if nargin<3; Winit = zeros(q,K-1);end % if there is no a specified initialization
lambda = 1e-6;% if a MAP regularization (a gaussian prior on W) (L_2 penalization); lambda isa positive hyperparameter

I = eye(q*(K-1));% define an identity matrix

%% IRLS Initialization (iter = 0)
W_old = Winit;

[piik_old, loglik_old] =multinomial_logit(W_old,X,Y);
loglik_old = loglik_old - lambda*(norm(W_old(:),2))^2;

iter = 0;
converge = 0;
max_iter = 300;
LL = [];

if verbose; fprintf(1,'IRLS : Iteration : %d "Log-lik": %f\n', iter, loglik_old);end
%% IRLS
while ~converge && (iter<max_iter)
    % Hw_old a squared matrix of dimensions  q*(K-1) x  q*(K-1)
    hx = q*(K-1);
    Hw_old = zeros(hx,hx);
    gw_old = zeros(q,K-1,1);
    
    % Gradient :
    for k=1:K-1
        gwk = Y(:,k) - piik_old(:,k);
        for qq=1:q
            vq = X(:,qq);
            gw_old(qq,k) = gwk'*vq;
        end
    end
    gw_old = reshape(gw_old,q*(K-1),1);
    % Hessian
    for k=1:K-1
        for ell=1:K-1
            delta_kl =(k==ell);% kronecker delta
            gwk = piik_old(:,k).*(ones(n,1)*delta_kl - piik_old(:,ell));
            Hkl = zeros(q,q);
            for qqa=1:q
                vqa=X(:,qqa);
                for qqb=1:q
                    vqb=X(:,qqb);
                    hwk = vqb'*(gwk.*vqa);
                    Hkl(qqa,qqb) = hwk;
                end
            end
            Hw_old((k-1)*q +1 : k*q, (ell-1)*q +1 : ell*q) = -Hkl;
        end
    end
    %% if a gaussien prior on W (lambda ~=0)
    Hw_old = Hw_old + lambda*I;
    gw_old = gw_old - lambda*W_old(:);
    %% Newton Raphson : W(c+1) = W(c) - H(W(c))^(-1)g(W(c))
    w = W_old(:) - inv(Hw_old)*gw_old ;%[(q+1)x(K-1),1]
    W = reshape(w,q,K-1);%[(q+1)*(K-1)]
    % mise a jour des probas et de la loglik
    [piik, loglik] =multinomial_logit(W,X,Y);
    loglik = loglik - lambda*(norm(W(:),2))^2;
    
    %% Verifier si Qw1(w^(t+1),w^(t))> Qw1(w^(t),w^(t))
    % (adaptive stepsize in case of troubles with stepsize 1) Newton Raphson : W(c+1) = W(c) - stepsize*H(W)^(-1)*g(W)
    stepsize = 1; % initialisation pas d'adaptation de l'algo Newton raphson
    alpha = 2;
    %ll = loglik_old;
    while (loglik < loglik_old)
        stepsize = stepsize/alpha; %
        %recalculate the parameter W and the "loglik"
        w = W_old(:) - stepsize* inv(Hw_old)*gw_old ;
        W = reshape(w,q,K-1);
        [piik, loglik] =multinomial_logit(W,X,Y);
        loglik = loglik - lambda*(norm(W(:),2))^2;
    end
    converge1 = abs((loglik-loglik_old)/loglik_old) <= 1e-7;
    converge2 = abs(loglik-loglik_old) <= 1e-6;
    
    converge = converge1 | converge2 ;
    
    piik_old = piik;
    W_old = W;
    iter = iter+1;
    LL = [LL loglik_old];
    loglik_old = loglik;
    if verbose
        fprintf(1,'IRLS : Iteration : %d "Log-lik": %f\n', iter, loglik_old);
    end
end % end of IRLS

if converge
    if verbose
        fprintf('\n');
        fprintf(1,'IRLS : convergence  OK ; nbr of iterations %d \n', iter);
        fprintf('\n');
    end
else
    fprintf('\n');
    fprintf('IRLS : doesn''t converged (augment the number of iterations > %d\n', max_iter) ;
end
% resultat
res.W = W;
res.LL= LL;
res.loglik = loglik;
res.piik = piik;

if lambda~=0 % calculate the value of the regularization part to calculate the value of the MAP criterion in case of regularization
    res.reg_irls = - lambda*(norm(W(:),2))^2; % bayesian l2 regularization
else
    res.reg_irls = 0;
end