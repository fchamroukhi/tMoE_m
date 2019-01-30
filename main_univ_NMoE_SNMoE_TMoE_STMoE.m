%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Matlab codes for univariate non-normal mixture of (linear) experts
% (NNMoE):
%
% 1. normal mixture of experts (NMoE)
% 2. skew-normal mixture (SNMoE)
% 3. >>>> t mixture of experts (TMoE) <<<<<
% 4. skew-t mixture of experts (STMoE)
% 5. Laplace mixture of experts (LMoE)
%
% C By Faicel Chamroukhi
% Written by Faicel Chamroukhi (may 2015)
% Updated by Faicel Chamroukhi (January 2016 : LMoE)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;


set(0,'defaultaxesfontsize',12);
%% model for data generation

%data_model = 'NMoE';
data_model = 'TMoE';
% data_model = 'SNMoE';
% data_model = 'STMoE';
% data_model = 'LMoE';

%% model for the inference
% inference_model = 'NMoE';
inference_model = 'TMoE';
% inference_model = 'SNMoE';
% inference_model = 'STMoE';
% inference_model = 'LMoE';
% inference_model = 'all';


%% EM features
nbr_EM_tries = 1;
max_iter_EM = 1500;
threshold = 1e-4;
verbose_EM = 1;
verbose_IRLS = 0;


%% MM features
nbr_MM_tries = 1;
max_iter_MM = 100;
threshold = 1e-5;
verbose_MM = 1;


n = 500;
%% model setting
K = 2;
p = 1;
q = 1;


Alphak = [0, 8]';
Betak = [0 0;
    -1 1];
Sigmak = [.1, .1];%the standard deviations
% Lambdak = [2, 3];
% Nuk = [5, 7];
Lambdak = [3, 5];
Nuk = [5, 7];

Zetak = [.1, .1]; % for the LMoE

x = linspace(-1, 1, n);

%%

WithOutliers = 0;
SaveResults  = 0;
%%
%% draw n samples from a chosen model :
switch data_model
    case 'NMoE'
        [y, klas, stats, Z] = sample_univ_NMoLE(Alphak, Betak, Sigmak, x);
    case 'SNMoE'
        [y, klas, stats, Z] = sample_univ_SNMoLE(Alphak, Betak, Sigmak, Lambdak, x);
    case 'TMoE'
        [y, klas, stats, Z] = sample_univ_TMoE(Alphak, Betak, Sigmak, Nuk, x);
    case 'STMoE'
        [y, klas, stats, Z] = sample_univ_STMoE(Alphak, Betak, Sigmak, Lambdak, Nuk, x);
    case 'LMoE'
        [y, klas, stats, Z] = sample_univ_LMoE(Alphak, Betak, Zetak, x);
    otherwise
        error('Unknown chosen model');
end


%% outliers
if WithOutliers
    rate = 0.05;%amount of outliers in the data
    No = round(length(y)*rate);
    outilers = -1.5 + 2*rand(No,1);
    tmp = randperm(length(y));
    Indout = tmp(1:No);
    y(Indout) = -2;%outilers;
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
    case 'LMoE'
        solution =  learn_univ_LMoE_MM(y, x, K, p, q, nbr_MM_tries, max_iter_MM, threshold, verbose_MM);
    case 'all'
        solution{1} =  learn_univ_NMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
        solution{2} =  learn_univ_SNMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
        solution{3} =  learn_univ_TMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
        solution{4} =  learn_univ_STMoE_EM(y, x, K, p, q, nbr_EM_tries, max_iter_EM, threshold, verbose_EM, verbose_IRLS);
        solution{5} =  learn_univ_LMoE_MM(y, x, K, p, q, nbr_MM_tries, max_iter_MM, threshold, verbose_MM, verbose_IRLS);
    otherwise
        error('Unknown chosen model');
end


%% plot of the results %%%%%%%%%%%%%%%%%%%%%%%

%show_results()

switch inference_model
    case 'all'
        %% data, True and Estimated mean functions
        figure,
        plot(x,y,'o','color',[0.6 0.6 .6])
        hold on, plot(x,stats.Ey,'k','linewidth',2)
        hold on, plot(x,solution{1}.Ey,'r--','linewidth',2)
        hold on, plot(x,solution{2}.Ey,'g-.','linewidth',2)
        hold on, plot(x,solution{3}.Ey,'b:','linewidth',2)
        hold on, plot(x,solution{4}.Ey,'c-.','linewidth',2)
        legend('data','True','NMoE', 'SNMoE', 'TMoE', 'STMoE');
        %ylim([-1.5, 1]);
        %% observed data log-likelihood
        figure,  plot(solution{1}.stored_loglik,'r-');
        hold on, plot(solution{2}.stored_loglik,'g-');
        hold on, plot(solution{3}.stored_loglik,'b-');
        hold on, plot(solution{4}.stored_loglik,'c-');
        xlabel('EM iteration number');
        ylabel('Observed data log-likelihood');
        legend('NMoE', 'SNMoE', 'TMoE', 'STMoE');
        box on;
    otherwise % a single model%% data, True and Estimated mean functions
        figure,
        h1 = plot(x,y,'o','color',[0.6 0.6 .6]);
        hold all
        h2 = plot(x,stats.Ey_k,'k--');
        h3 = plot(x,solution.Ey_k,'r--');
        
        h4 = plot(x,stats.Ey,'k','linewidth',2);
        h5 = plot(x,solution.Ey,'r','linewidth',2);
        xlabel('x'), ylabel('y');
        ylim([-2, 1])
        hold off
        legend([h1, h4, h3(1), h5, h2(1)], 'data',['True mean',' (',data_model,')'],'True Experts', ...
            ['Estimated mean',' (',inference_model,')'],'Estimated Experts',...
            'Location','SouthWest');
        legend('boxoff')
        
        
        %         %% data, True and Estimated expert components and gating functions
        %         figure,
        %         plot(x,y,'o','color',[0.6 0.6 .6])
        %         hold all
        %         % true experts components
        %         plot(x,stats.Ey_k);
        %         % estimated experts components
        %         plot(x,solution.Ey_k,'linewidth',2);
        %         xlabel('x'), ylabel('y');
        %         ylim([-2, 1])
        %         hold off
        %legend('data','True Experts','Estimated Experts');
        % Estimated mixing probabilities
        
        color = {'k','r','b','g','m','c','y',   'k','r','b','g','m','c','y',    'k','r','b','g','m','c','y'};
        figure,
        for k=1:K
            plot(x,solution.Piik(:,k),[color{k},'-'],'linewidth',2);
            hold on;
        end
        hold off
        %plot(x,solution.Piik,'linewidth',2);
        ylim([0, 1]);
        xlabel('x'), ylabel('Mixing probabilities');
        
        
        %% data, True and Estimated mean functions and ? 2*sigma confidence regions
        figure,
        h1 = plot(x,y,'o','color',[0.6 0.6 .6]);
        hold all
        % true
        h2 = plot(x,stats.Ey,'k','linewidth',1.5);
        h3 = plot(x,[stats.Ey-2*sqrt(stats.Vary), stats.Ey+2*sqrt(stats.Vary)],'k--','linewidth',1);
        %hold on, shadedErrorBar(x,stats.Ey,2*sqrt(stats.Vary),'r');
        % estimated
        h4 = plot(x,solution.Ey,'r','linewidth',2);
        h5 = plot(x,[solution.Ey-2*sqrt(solution.Vary), solution.Ey+2*sqrt(solution.Vary)],'r--','linewidth',1);
        legend([h1, h2, h3(1), h4, h5(1)],'data', ['True mean',' (',data_model,')'], ....
            'True conf. regions',['Estimated mean',' (',inference_model,')'], 'Estimated conf. regions',...
            'Location','SouthWest');
        legend('boxoff');
        xlabel('x'), ylabel('y');
        ylim([-2, 1])
        hold off
        
        %% obtained partition
        figure,
        hold all
        % true partiton
        h11 = plot(x,stats.Ey_k(:,1),'k','linewidth',1.2);
        h12 = plot(x,stats.Ey_k(:,2),'r','linewidth',1.2);
        %h13 = gscatter(x,y,klas,'kr','oo');
        h13 =  plot(x(klas==1),y(klas==1),'o','color',[0.6 0.6 .6]);
        h14 =  plot(x(klas==2),y(klas==2),'o','color',[1 0 0]);
        ylim([-2, 1])
        legend([h11, h12, h13, h14],'Actual expert mean 1','Actual expert mean 2','Actual cluster 1','Actual cluster 2','Location','SouthWest');
        box on
        xlabel('x'), ylabel('y');
        
        legend('boxoff')
        hold off
        
        figure
        hold all
        % estimated partition
        h21 = plot(x,solution.Ey_k(:,1),'k','linewidth',1.2);
        h22 = plot(x,solution.Ey_k(:,2),'r','linewidth',1.2);
        %h23 = gscatter(x,y,solution.klas,'kr','oo');
        h23 =  plot(x(solution.klas==1),y(solution.klas==1),'o', 'color',[0.6 0.6 .6]);
        h24 =  plot(x(solution.klas==2),y(solution.klas==2),'o', 'color',[1 0 0]);
        legend([h21, h22, h23, h24],'Estimated expert mean 1','Estimated expert mean 2','Estimated cluster 1','Estimated cluster 2','Location','SouthWest');
        ylim([-2, 1]);
        box on
        xlabel('x'), ylabel('y');
        
        legend('boxoff')
        hold off
        %% observed data log-likelihood
        figure, plot(solution.stored_loglik,'-');
        xlabel('EM iteration number');
        ylabel('Observed data log-likelihood');
        legend([inference_model, ' log-likelihood']);
        legend('boxoff')
        box on;
end

% % %% some measure of fit (MSE between the true and the estimated
% mean functions)

% approxim error
MSE = (1/n)*sum((solution.Ey - stats.Ey).^2)


%% save the results
% if SaveResults
%     if WithOutliers
%         saveDirectory = ['./Results/Simulation-1-with-Outliers/',data_model];
%         eval(['cd ', saveDirectory]);
%         % save the data
%         eval(['save y_',data_model,'_outliers y']);
%         eval(['save klas_',data_model,'_outliers klas']);
%         eval(['save stats_',data_model,'_outliers stats']);
%         eval(['save Z_',data_model,'_outliers Z']);
%         % save the solution
%         eval(['save sol_',data_model,'_noisy_',inference_model,' solution']);
%         % save the figures
%         saveas(1,['TwoClust-Outliers-',data_model,'_',inference_model,'_Means'],'epsc')
%         saveas(2,['TwoClust-Outliers-',data_model,'_',inference_model,'_MixingProb'],'epsc')
%         saveas(3,['TwoClust-Outliers-',data_model,'_',inference_model,'_Means_ConfidRegions'],'epsc')
%         saveas(4,['TwoClust-Outliers-',data_model,'_',inference_model,'_TruePartition'],'epsc')
%         saveas(5,['TwoClust-Outliers-',data_model,'_',inference_model,'_EstimatedPartition'],'epsc')
%         saveas(6,['TwoClust-Outliers-',data_model,'_',inference_model,'_Loglik'],'epsc')
%     else
%         saveDirectory = ['./Results/Simulation-1/',data_model];
%         eval(['cd ', saveDirectory]);
%         % save the data
%         eval(['save y_',data_model,' y']);
%         % save the solution
%         eval(['save klas_',data_model,' klas']);
%         eval(['save stats_',data_model,' stats']);
%         eval(['save Z_',data_model,' Z']);
%         % save the solution
%         eval(['save sol_',data_model,'_',inference_model,' solution']);
%         % save the figures
%         saveas(1,['TwoClust-',data_model,'_',inference_model,'_Means'],'epsc')
%         saveas(2,['TwoClust-',data_model,'_',inference_model,'_MixingProb'],'epsc')
%         saveas(3,['TwoClust-',data_model,'_',inference_model,'_Means_ConfidRegions'],'epsc')
%         saveas(4,['TwoClust-',data_model,'_',inference_model,'_TruePartition'],'epsc')
%         saveas(5,['TwoClust-',data_model,'_',inference_model,'_EstimatedPartition'],'epsc')
%         saveas(6,['TwoClust-',data_model,'_',inference_model,'_Loglik'],'epsc')
%     end
% end




% %% use the model for some fixed (real) data
% if strcmp(data_model,'Tone')
%     data = xlsread('./Results/Real-Data/Tone/Tone.xlsx');
%     x = data(:,1);
%     y = data(:,2);
%     
%     if (WithOutilers)
%         %     rate = 0.1;
%         %     No = round(length(y)*rate);
%         %     outilers = -1.5 + 2*rand(No,1);
%         %     tmp = randperm(length(y));
%         %     Indout = tmp(1:No);
%         %     y(Indout) = -.1;%outilers;
%         % end
%         y   = [y; 4*ones(10,1)];
%         x   = [x; 0*ones(10,1)];
%         %             tmp = randperm(length(y));
%         %             x   = x(tmp);
%         %             y   = y(tmp);
%     end
% end




