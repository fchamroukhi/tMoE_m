function [y] = sample_univ_T(mu, sigma, nu, n)
% draw n samples from a univariate Student t distribution
%
% mu: location parameter
% sigma: scale parameter
% nu: degrees of freedom parameter
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin==3, n= length(mu); end
% y = zeros(n,1);
% for i=1:n
%    %% by using the hierarchical representation of the t distribution
% %    Wi = gamrnd(nu/2,2/nu); %or by equivalence chi2rnd(nuk)/nuk;
% %    y(i) = normrnd(mu, sqrt(sigma^2/Wi));
%    
%    %% or by equivalence by using the stochastic representation of the t-distribution   
%     Wi = gamrnd(nu/2,2/nu); %or by equivalence chi2rnd(nuk)/nuk;
%     Ei = normrnd(0,1);
%     y(i) = mu + sigma*Ei/sqrt(Wi);
%     %% or by using the Matlab function for drawing samples the standard t
% %     distribution (trnd)
% %     y(i) = mu + sigma*trnd(nu);
%      %% another way
% %    y(i) = normrnd(mu, sigma)*sqrt(nu/(2*randg(nu/2,1)));
% end

% vectoraial form

%% by using the hierarchical representation of the t distribution
% W = gamrnd(nu/2,2/nu, n, 1); %or by equivalence chi2rnd(nuk)/nuk;
% y = normrnd(mu, sqrt(sigma^2/W), n, 1);

%% or by equivalence by using the stochastic representation of the t-distribution
W = gamrnd(nu/2,2./nu, n, 1); %or by equivalence chi2rnd(nuk)/nuk;
E = normrnd(0,1, n, 1);
y = mu + sigma*E./sqrt(W);
%% or by using the Matlab function for drawing samples the standard t distribution (trnd)
% y = mu + sigma.*trnd(nu);
%% another way
% y = normrnd(mu, sigma, n, 1).*sqrt(nu./(2*randg(nu/2, n, 1)));

%stats
