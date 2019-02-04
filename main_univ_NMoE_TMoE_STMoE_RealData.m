%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Matlab/octave toolbox for univariate non-normal mixture of (linear) experts
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
data_model = 'Tone'; K = 2; p = 1; q = 1;
data_model = 'TemperatureAnomaly'; K = 2; p = 1; q = 1;


%% model for the inference
 inference_model = 'NMoE';
%inference_model = 'TMoE';
% inference_model = 'SNMoE';
% inference_model = 'STMoE';
% inference_model = 'LMoE';
% inference_model = 'all';


%% EM features
nbr_EM_tries = 1;
max_iter_EM = 1500;
threshold = 1e-5;
verbose_EM = 1;
verbose_IRLS = 0;

%% MM features
nbr_MM_tries = 1;
max_iter_MM = 800;
threshold = 1e-5;
verbose_MM = 1;

%
WithOutilers = 1;
SaveResults  = 0;

%% Tone data set
if strcmp(data_model,'Tone')
    data = xlsread('./Results/Real-Data/Tone/Tone.xlsx');
    x = data(:,1);
    y = data(:,2);
    
    if (WithOutilers)
        %     rate = 0.1;
        %     No = round(length(y)*rate);
        %     outilers = -1.5 + 2*rand(No,1);
        %     tmp = randperm(length(y));
        %     Indout = tmp(1:No);
        %     y(Indout) = -.1;%outilers;
        % end
        y   = [y; 4*ones(10,1)];
        x   = [x; 0*ones(10,1)];
                    tmp = randperm(length(y));
                    x   = x(tmp);
                    y   = y(tmp);
    end
end
%% Temperature Anomaly
if strcmp(data_model,'TemperatureAnomaly')
    load 'data/TemperatureAnomaly';
    x = TemperatureAnomaly(:,1);%(3:end-2,1); % if the values for 1880 1881, 2013 and 2014 are not included (only from 1882-2012)
    y = TemperatureAnomaly(:,2);%(3:end-2,2); % if the values for 1880 1881, 2013 and 2014 are not included (only from 1882-2012)
end

% %% Sinc
% if strcmp(data_model,'Sinc')
%     x=linspace(-1,1,n);
%     stats.Ey = sinc(pi*x)';
%     y = stats.Ey + .1*randn(n,1);
% end


%% save the data
if SaveResults
    if WithOutilers
        saveDirectory = ['./Results/Real-Data-with-Outliers/',data_model];
        eval(['cd ', saveDirectory]);
        % save the data
        eval(['save x_',data_model,'_outliers x']);
        eval(['save y_',data_model,'_outliers y']);
    else
        saveDirectory = ['./Results/Real-Data/',data_model];
        eval(['cd ', saveDirectory]);
        % save the data
        eval(['save x_',data_model,' x']);
        eval(['save y_',data_model,' y']);
    end
    cd '~/Desktop/Mixtures-Non-Normals/Codes-NNMoE/codes-non-normal-MoE';
end
% % noisy switch
% load y
% load x

figure,
plot(x, y, 'o')
xlabel('x')
ylabel('y')
title([data_model,' data set'])


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
    case 'LMoE'
        solution =  learn_univ_LMoE_MM(y, x, K, p, q, nbr_MM_tries, max_iter_MM, threshold, verbose_MM);
    case 'all'
        solution{1} =  learn_univ_NMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
        solution{2} =  learn_univ_SNMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
        solution{3} =  learn_univ_TMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
        solution{4} =  learn_univ_STMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
        solution{5} =  learn_univ_LMoE_MM(y, x, K, p, q, nbr_MM_tries, max_iter_MM, threshold, verbose_MM);
    otherwise
        error('Unknown chosen model');
end

figure, plot(y,'ko'),
hold on, plot(solution.Ey_k,'--')
hold on, plot(solution.Ey,'r-','linewidth',2)
%set(gca,'XtickLabel',[1:length(x)])
xlabel('t')
ylabel('power (W)')
ylim([200 500])

figure,plot(solution.Piik,'linewidth',2)
xlabel('t')
ylabel('Gating Network')

% save y y
% save x x
% cd '~/Desktop/Talk-ERCIM-2015/Figures/'
% saveas(gca,['Switch-',inference_model],'epsc')

%% plot of the results %%%%%%%%%%%%%%%%%%%%%%%
if strcmp(data_model,'Tone')
    [x, indx] = sort(x);
    y = y(indx);
    solution.Ey_k = solution.Ey_k(indx,:);
    solution.Piik = solution.Piik(indx,:);
    solution.Ey = solution.Ey(indx);
    solution.Vary = solution.Vary(indx);
    solution.klas = solution.klas(indx);
    
    % Estiamted mixing probabilities
    figure, title(inference_model)
    set(gca,'YTick',0:.2:1);
    hold all
    plot(x,solution.Piik(:,1),'k-','linewidth',1.2);
    plot(x,solution.Piik(:,2),'r-','linewidth',1.2);
    box on
    ylim([0 1]);
    
    xlim([min(x), max(x)]);
    xlabel('x'), ylabel('Mixing probabilities');
    hold off
    
    figure,title(inference_model)
    hold all,
    % Estimated partition and means
    h1 = gscatter(x,y,solution.klas,'kr','oo');
    h21 = plot(x,solution.Ey_k(:,1),'k-','linewidth',1);
    h22 = plot(x,solution.Ey_k(:,2),'r-','linewidth',1);
    %
    ylim([min(y), max(y)]);
    xlim([min(x), max(x)]);
    %     if strcmp(data_model,'Tone'), set(gca,'YTick',0:0.2:1);end
    box on;
    legend([h1(1), h1(2), h21, h22],'Cluster 1','Cluster 2', ...
        'Expert mean 1','Expert mean 2','Location','NorthWest');
    legend('boxoff')
    hold off
    %% data and estimated mean function
    figure,  title(inference_model)
    hold on, plot(x,y, 'o', 'color',[0.6 0.6 .6])
    hold on, plot(x,solution.Ey,'r','linewidth',1.5)
    % estimated empirical confidence regions
    hold on, plot(x,[solution.Ey-2*sqrt(solution.Vary), ...
        solution.Ey+2*sqrt(solution.Vary)],'r--','linewidth',1);
    ylim([min(y), max(y)]);
    xlim([min(x), max(x)]);
    %     if strcmp(data_model,'Bishop'), set(gca,'YTick',0:0.2:1);end
    xlabel('x'), ylabel('y');
    legend('data',['Estimated mean',' (',inference_model,')'],'Location','NorthWest');
    legend('boxoff')
    box on;
    %% observed data log-likelihood
    figure, title(inference_model),
    hold on, plot(solution.stored_loglik,'-');
    xlabel('EM iteration number');
    ylabel('Observed data log-likelihood');
    legend([inference_model, ' log-likelihood']);
    legend('boxoff')
    box on;
end

%% Temperature anomaly
if strcmp(data_model,'TemperatureAnomaly')
    [x, indx] = sort(x);
    y = y(indx);
    solution.Ey_k = solution.Ey_k(indx,:);
    solution.Piik = solution.Piik(indx,:);
    solution.Ey = solution.Ey(indx);
    solution.Vary = solution.Vary(indx);
    solution.klas = solution.klas(indx);
    
    % Estiamted mixing probabilities
    figure, title(inference_model)
    hold all
    plot(x,solution.Piik(:,1),'k-','linewidth',1.2);
    plot(x,solution.Piik(:,2),'r-','linewidth',1.2);
    box on
    ylim([0 1]);
    set(gca,'YTick',0:.2:1);
    
    xlim([1870, 2018]);
    xlabel('x'), ylabel('Mixing probabilities');
    hold off
    
    figure,title(inference_model)
    hold all,
    % Estimated partition and means
    h1 = gscatter(x,y,solution.klas,'kr','oo');
    h21 = plot(x,solution.Ey_k(:,1),'k-','linewidth',1);
    h22 = plot(x,solution.Ey_k(:,2),'r-','linewidth',1);
    %
    ylim([-0.7, 1]);
    xlim([1870, 2018]);
    
    set(gca,'YTick',[-0.5 0 0.5 1]);
    
    %     if strcmp(data_model,'Tone'), set(gca,'YTick',0:0.2:1);end
    box on;
    legend([h1(1), h1(2), h21, h22],'Cluster 1','Cluster 2', ...
        'Expert mean 1','Expert mean 2','Location','NorthWest');
    legend('boxoff')
    hold off
    %% data and estimated mean function
    figure,  title(inference_model)
    hold on, plot(x,y,'o', 'color',[0.6 0.6 .6])
    hold on, plot(x,solution.Ey,'r','linewidth',1.5)
    % estimated empirical confidence regions
    hold on, plot(x,[solution.Ey-2*sqrt(solution.Vary), ...
        solution.Ey+2*sqrt(solution.Vary)],'r--','linewidth',1);
    ylim([-0.7, 1]);
    xlim([1870, 2018]);
    set(gca,'YTick',[-0.5 0 0.5 1]);
    
    %     if strcmp(data_model,'Bishop'), set(gca,'YTick',0:0.2:1);end
    xlabel('x'), ylabel('y');
    legend('data',['Estimated mean',' (',inference_model,')'],'Location','NorthWest');
    legend('boxoff')
    box on;
    %% observed data log-likelihood
    figure, title(inference_model),
    hold on, plot(solution.stored_loglik,'-');
    xlabel('EM iteration number');
    ylabel('Observed data log-likelihood');
    legend([inference_model, ' log-likelihood']);
    legend('boxoff')
    box on;
end


%% Motorcycle
if strcmp(data_model,'Motorcycle')
    %     [x, indx] = sort(x);
    %     y = y(indx);
    %     solution.Ey_k = solution.Ey_k(indx,:);
    %     solution.Piik = solution.Piik(indx,:);
    %     solution.Ey = solution.Ey(indx);
    %     solution.Vary = solution.Vary(indx);
    %     solution.klas = solution.klas(indx);
    
    % Estiamted mixing probabilities
    figure, title(inference_model)
    hold all
    plot(x,solution.Piik,'linewidth',1.2);
    box on
    ylim([0 1]);
    xlim([0, 60]);
    
    set(gca,'YTick',0:.2:1);
    
    %     xlim([ ]);
    xlabel('x'), ylabel('Mixing probabilities');
    hold off
    
    figure,title(inference_model)
    hold all,
    % Estimated partition and means
    h1 = gscatter(x,y,solution.klas);
    h21 = plot(x,solution.Ey_k,'linewidth',1);
    ylim([-150, 100]);
    xlim([0, 60]);
    %     set(gca,'YTick',[-0.5 0 0.5 1]);
    
    %     if strcmp(data_model,'Tone'), set(gca,'YTick',0:0.2:1);end
    box on;
    %     legend([h1(1), h1(2), h21, h22],'Cluster 1','Cluster 2', ...
    %         'Expert mean 1','Expert mean 2','Location','NorthWest');
    %     legend('boxoff')
    hold off
    %% data and estimated mean function
    figure,  title(inference_model)
    hold on, plot(x,y,'o', 'color',[0.6 0.6 .6])
    hold on, plot(x,solution.Ey,'r','linewidth',1.5)
    % estimated empirical confidence regions
    hold on, plot(x,[solution.Ey-2*sqrt(solution.Vary), ...
        solution.Ey+2*sqrt(solution.Vary)],'r--','linewidth',1);
    ylim([-150, 100]);
    xlim([0, 60]);
    %     set(gca,'YTick',[-0.5 0 0.5 1]);
    
    xlabel('x'), ylabel('y');
    legend('data',['Estimated mean',' (',inference_model,')'],'Location','NorthWest');
    legend('boxoff')
    box on;
    %% observed data log-likelihood
    figure, title(inference_model),
    hold on, plot(solution.stored_loglik,'-');
    xlabel('EM iteration number');
    ylabel('Observed data log-likelihood');
    legend([inference_model, ' log-likelihood']);
    legend('boxoff')
    box on;
end


%% save the results
if SaveResults
    if WithOutilers
        saveDirectory = ['./Results/Real-Data-with-Outliers/',data_model];
        eval(['cd ', saveDirectory]);
        % save the solution
        eval(['save sol_',data_model,'_noisy_',inference_model,' solution']);
        % save the figures
        saveas(1,[data_model,'_data_Outliers_',inference_model,'_MixingProb'],'epsc')
        saveas(2,[data_model,'_data_Outliers_',inference_model,'_EstimatedPartition'],'epsc')
        saveas(3,[data_model,'_data_Outliers_',inference_model,'_Means_ConfidRegions'],'epsc')
        saveas(4,[data_model,'_data_Outliers_',inference_model,'_Loglik'],'epsc')
    else
        saveDirectory = ['./Results/Real-Data/',data_model];
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







