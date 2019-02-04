%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Matlab codes for univariate non-normal mixture of (linear) experts
% (NNMoE):
%
% 1. normal mixture of experts (NMoE)
% 2. skew-normal mixture (SNMoE)
% 3. t mixture of experts (TMoE)
% 4. skew-t mixture of experts (STMoE)
%
% C By Faicel Chamroukhi
% Written By Faicel Chamroukhi (may 2015)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;


set(0,'defaultaxesfontsize',14);
%% model for data generation
data_model = 'Bishop'; K = 3; p = 1; q = 1;
  data_model = 'Sinc'; K = 3; p = 4; q = 1;
 data_model =  'Motorcycle'; K = 3; p = 4; q = 1;

% data BM3E, motocycles etc

%% model for the inference
%inference_model = 'NMoE';
inference_model = 'TMoE';
% inference_model = 'SNMoE';
% inference_model = 'STMoE';
% inference_model = 'all';


%% EM features
nbr_EM_tries = 1;
max_iter_EM = 1500;
threshold = 1e-6;
verbose_EM = 1;
verbose_IRLS = 0;

n = 200;

NewSample    = 1;

WithOutilers = 0;
SaveResults  = 0;

if NewSample
    %% draw a sample :
    %% Bishop toy data set
    if strcmp(data_model,'Bishop')
        [y, x] = sample_dataBM3E(n);
    end
    %% Sinc
    if strcmp(data_model,'Sinc')
        x=linspace(-1,1,n);
        stats.Ey = sinc(pi*x)';
        y = stats.Ey + .1*randn(n,1);
    end
    if strcmp(data_model,'Motorcycle')
        load data/motorcycle;
        x= motorcycle.x;
        y =motorcycle.y;
        y=zscore(y);
    end
else
    if WithOutilers
        % load the data
        cmdx = ['load ./Results/Simulation-2-with-Outliers/',data_model,'/x_',data_model,'_outliers'];
        eval(cmdx);
        cmdy = ['load ./Results/Simulation-2-with-Outliers/',data_model,'/y_',data_model,'_outliers'];
        eval(cmdy);
    else
        % load the data
        cmdx = ['load ./Results/Simulation-2/',data_model,'/x_',data_model];
        eval(cmdx);
        cmdy = ['load ./Results/Simulation-2/',data_model,'/y_',data_model];
        eval(cmdy);
    end
end

%% outliers
if (WithOutilers && NewSample)
    %     rate = 0.1;
    %     No = round(length(y)*rate);
    %     outilers = -1.5 + 2*rand(No,1);
    %     tmp = randperm(length(y));
    %     Indout = tmp(1:No);
    %     y(Indout) = outilers;
    Nout = 50;
    x = [x; rand(Nout,1)];
    y = [y; zeros(Nout,1)];
    tmp = randperm(length(y));
    x=x(tmp);
    y=y(tmp);
end



%% save the data
if SaveResults
    if WithOutilers
        saveDirectory = ['./Results/Simulation-2-with-Outliers/',data_model];
        eval(['cd ', saveDirectory]);
        % save the data
        eval(['save x_',data_model,'_outliers x']);
        eval(['save y_',data_model,'_outliers y']);
        % save the solution
        eval(['save sol_',data_model,'_noisy_',inference_model,' solution']);
        % save the figures
        % saveas(
    else
        saveDirectory = ['./Results/Simulation-2/',data_model];
        eval(['cd ', saveDirectory]);
        % save the data
        eval(['save x_',data_model,' x']);
        eval(['save y_',data_model,' y']);
    end
    cd '~/Desktop/Mixtures-Non-Normals/Codes-NNMoE/codes-non-normal-MoE';
end

%% learn a chosen model from the sampled data
switch inference_model
    case 'NMoE'
        solution =  learn_univ_NMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
    case 'SNMoE'
        solution =  learn_univ_SNMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
    case 'TMoE'
        solution =  learn_univ_TMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
    case 'STMoE'
        solution =  learn_univ_STMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
    case 'all'
        solution{1} =  learn_univ_NMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
        solution{2} =  learn_univ_SNMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
        solution{3} =  learn_univ_TMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
        solution{4} =  learn_univ_STMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
    otherwise
        error('Unknown chosen model');
end

if strcmp(data_model,'Bishop')
    [x, indx] = sort(x);
    y = y(indx);
    solution.Ey_k = solution.Ey_k(indx,:);
    solution.Piik = solution.Piik(indx,:);
    solution.Ey = solution.Ey(indx);
    solution.klas = solution.klas(indx);
end

%% plot of the results %%%%%%%%%%%%%%%%%%%%%%%
% Estiamted mixing probabilities
figure,title(inference_model);

hold all
plot(x,solution.Piik(:,1),'k-','linewidth',1.2);
plot(x,solution.Piik(:,2),'r-','linewidth',1.2);
plot(x,solution.Piik(:,3),'b-','linewidth',1.2);
box on
ylim([0 1]);
set(gca,'YTick',0:.2:1)

xlim([min(x), max(x)]);
xlabel('x'), ylabel('Mixing probabilities');
hold off

figure,title(inference_model)
hold all,
% Estimated partition and means
h1 = gscatter(x,y,solution.klas,'krb','ooo');
h21 = plot(x,solution.Ey_k(:,1),'k-','linewidth',1);
h22 = plot(x,solution.Ey_k(:,2),'r-','linewidth',1);
h23 = plot(x,solution.Ey_k(:,3),'b-','linewidth',1);
%
ylim([min(y), max(y)]);
xlim([min(x), max(x)]);
if strcmp(data_model,'Bishop'), set(gca,'YTick',0:0.2:1);end
box on;
h=legend([h1(1), h1(2), h1(3), h21, h22, h23],'Cluster 1','Cluster 2', 'Cluster 3',...
    'Expert mean 1','Expert mean 2','Expert mean 3');%,'Location','MiddleEast');
rect = [0.72, 0.6, .1, .1];
set(h, 'Position', rect)
legend('boxoff')
hold off
%% data and estimated mean function
figure,  title(inference_model);
hold on, plot(x,y,'o', 'color',[0.6 0.6 .6])
hold on, plot(x,solution.Ey,'r','linewidth',1.5)
% estimated empirical confidence regions
hold on, plot(x,[solution.Ey(indx)-2*sqrt(solution.Vary(indx)), ...
    solution.Ey(indx)+2*sqrt(solution.Vary(indx))],'r--','linewidth',1);
ylim([min(y), max(y)]);
xlim([min(x), max(x)]);
if strcmp(data_model,'Bishop'), set(gca,'YTick',0:0.2:1);end
xlabel('x'), ylabel('y');
legend('data',['Estimated mean',' (',inference_model,')'],'Location','NorthWest');
legend('boxoff')
box on;
%% observed data log-likelihood
figure, title(inference_model);
hold on, plot(solution.stored_loglik,'-');
xlabel('EM iteration number');
ylabel('Observed data log-likelihood');
legend([inference_model, ' log-likelihood']);
legend('boxoff')
box on;

% end

%% save the results
if SaveResults
    if WithOutilers
        saveDirectory = ['./Results/Simulation-2-with-Outliers/',data_model];
        eval(['cd ', saveDirectory]);
        % save the solution
        eval(['save sol_',data_model,'_noisy_',inference_model,' solution']);
        % save the figures
        % saveas(
    else
        saveDirectory = ['./Results/Simulation-2/',data_model];
        eval(['cd ', saveDirectory]);
        % save the solution
        eval(['save sol_',data_model,'_',inference_model,' solution']);
        % save the figures
        saveas(1,[data_model,'_data_',inference_model,'_MixingProb'],'epsc')
        saveas(2,[data_model,'_data_',inference_model,'_EstimatedPartition'],'epsc')
        saveas(3,[data_model,'_data_',inference_model,'_Means_ConfidRegions'],'epsc')
        saveas(4,[data_model,'_data_',inference_model,'_Loglik'],'epsc')
    end
    cd '~/Desktop/Mixtures-Non-Normals/Codes-NNMoE/codes-non-normal-MoE';
end







