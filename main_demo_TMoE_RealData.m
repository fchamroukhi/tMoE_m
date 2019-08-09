%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% >>t mixture of experts (TMoE): Robust mixtures-of-experts modeling using the t distribution <<
%
% TMoE : A Matlab/Octave toolbox for modeling, sampling, inference, regression and clustering of
% heterogeneous data with the t Mixture-of-Experts (TMoE) model.
%
% TMoE provides a flexible and robust modeling framework for heterogenous data with possibly
% heavy-tailed distributions and corrupted by atypical observations. TMoE consists of a mixture of K
% t expert regressors network (of degree p) gated by a softmax gating network (with regression
% degree q) and is represented by
% - The gating net. parameters $\alpha$'s of the softmax net.
% - The experts network parameters: The location parameters (regression coefficients) $\beta$'s,
% scale parameters $\sigma$'s, and the degree of freedom (robustness) parameters $\nu$'s.
%
% TMoE thus generalises  mixtures of (normal, t, and) distributions and mixtures of regressions with
% these distributions. For example, when $q=0$, we retrieve mixtures of (t-, or normal) regressions,
% and when both $p=0$ and $q=0$, it is a mixture of (t-, or normal) distributions. It also reduces
% to the standard (normal, t) distribution when we only use a single expert (K=1).
%
% Model estimation/learning is performed by a dedicated expectation conditional maximization (ECM)
% algorithm by maximizing the observed data log-likelihood. We provide simulated examples to
% illustrate the use of the model in model-based clustering of heterogeneous regression data and in
% fitting non-linear regression functions. Real-world data examples of tone perception for musical
% data analysis, and the one of temperature anomalies for the analysis of climate change data, are
% also provided as application of the model.
%
% To run it on the provided examples, please run "main_demo_TMoE_SimulatedData.m" or
% "main_demo_TMoE_RealData.m"
%
%% Please cite the code and the following papers when using this code:
% - F. Chamroukhi. Robust mixture of experts modeling using the $t$-distribution. Neural Networks, V. 79, p:20?36, 2016
% - F. Chamroukhi. Non-Normal Mixtures of Experts. arXiv:1506.06707, July, 2015
%
% (c) Introduced and written by Faicel Chamroukhi (may 2015)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;


set(0,'defaultaxesfontsize',14);
%%  chose a real data and some model structure
data_set = 'Tone'; K = 2; p = 1; q = 1;
data_set = 'TemperatureAnomaly'; K = 2; p = 1; q = 1;
% data_set = 'motorcycle'; K = 4; p = 2; q = 1;

%% EM options
nbr_EM_tries = 2;
max_iter_EM = 1500;
threshold = 1e-6;
verbose_EM = 1;
verbose_IRLS = 0;

switch data_set
    %% Tone data set
    case 'Tone'
        data = xlsread('data/Tone.xlsx');
        x = data(:,1);
        y = data(:,2);
        %% Temperature Anomaly
    case 'TemperatureAnomaly'
        load 'data/TemperatureAnomaly';
        x = TemperatureAnomaly(:,1);%(3:end-2,1); % if the values for 1880 1881, 2013 and 2014 are not included (only from 1882-2012)
        y = TemperatureAnomaly(:,2);%(3:end-2,2); % if the values for 1880 1881, 2013 and 2014 are not included (only from 1882-2012)
        %% Motorcycle
    case 'motorcycle'
        load 'data/motorcycle.mat';
        x=motorcycle.x;
        y=motorcycle.y;
    otherwise
        data = xlsread('data/Tone.xlsx');
        x = data(:,1);
        y = data(:,2);
end
figure,
plot(x, y, 'ko')
xlabel('x')
ylabel('y')
title([data_set,' data set'])

%% learn the model from the  data

TMoE =  learn_TMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);

disp('- fit completed --')

show_TMoE_results(x, y, TMoE)
% Note that as it uses the t distribution, so the mean and the variance might be not defined (if Nu <1 and or <2), and hence the
% mean functions and confidence regions might be not displayed..



