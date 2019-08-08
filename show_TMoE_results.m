function show_TMoE_results(x, y, TMoE, klas, TrueStats)

set(0,'defaultaxesfontsize',12);
color = {'k','r','b','g','m','c','y',   'k','r','b','g','m','c','y',    'k','r','b','g','m','c','y'};
yaxislim = [min(y)-std(y), max(y)+std(y)];

param = TMoE.param;
stats = TMoE;%.stats;
K = length(param.Nuk);

if nargin>3
    figure,
    h1 = plot(x,y,'o','color',[0.6 0.6 .6]);
    hold all
    h2 = plot(x,TrueStats.Ey_k,'k--');
    h3 = plot(x,stats.Ey_k,'r--');
    
    h4 = plot(x,TrueStats.Ey,'k','linewidth',2);
    h5 = plot(x,stats.Ey,'r','linewidth',2);
    xlabel('x'), ylabel('y');
    ylim(yaxislim)
    hold off
    legend([h1, h4, h3(1), h5, h2(1)], 'data',['True mean',' (TMoE)'],'True Experts', ...
        ['Estimated mean',' (TMoE)'],'Estimated Experts',...
        'Location','SouthWest');
    legend('boxoff')
    figure,
    for k=1:K
        plot(x,stats.Piik(:,k),[color{k},'-'],'linewidth',2);
        hold on;
    end
    hold off
    ylim([0, 1]);
    xlabel('x'), ylabel('Gating network probabilities');
    
    
    %% data, True and Estimated mean functions and pointwise 2*sigma confidence regions
    figure,
    h1 = plot(x,y,'o','color',[0.6 0.6 .6]);
    hold all
    % true
    h2 = plot(x,TrueStats.Ey,'k','linewidth',1.5);
    h3 = plot(x,[TrueStats.Ey-2*sqrt(TrueStats.Vy), TrueStats.Ey+2*sqrt(TrueStats.Vy)],'k--','linewidth',1);
    % estimated
    h4 = plot(x,stats.Ey,'r','linewidth',2);
    h5 = plot(x,[stats.Ey-2*sqrt(stats.Vy), stats.Ey+2*sqrt(stats.Vy)],'r--','linewidth',1);
    legend([h1, h2, h3(1), h4, h5(1)],'data', ['True mean',' (TMoE)'], ....
        'True conf. regions',['Estimated mean',' (TMoE)'], 'Estimated conf. regions',...
        'Location','SouthWest');
    legend('boxoff');
    xlabel('x'), ylabel('y');
    ylim(yaxislim)
    hold off
    
    %% obtained partition
    figure,
    hold all
    % true partiton
    for k=1:max(klas)
        plot(x,TrueStats.Ey_k(:,k),color{k},'linewidth',1.2);
        plot(x(klas==k),y(klas==k),[color{k},'o']);
    end
    legend('True expert means','True clusters');
    
    ylim(yaxislim)
    box on
    xlabel('x'), ylabel('y');
    
    legend('boxoff')
    hold off
    
    figure
    hold all
    % estimated partition
    for k=1:K
        plot(x,stats.Ey_k(:,k),color{k},'linewidth',1.2);
        plot(x(stats.klas==k),y(stats.klas==k),[color{k},'o']);
    end
    legend('Estimated expert means','Estimated clusters');    ylim(yaxislim)
    box on
    xlabel('x'), ylabel('y');
    
    legend('boxoff')
    hold off
    %% observed data log-likelihood
    figure, plot(stats.stored_loglik,'-');
    xlabel('EM iteration number');
    ylabel('Observed data log-likelihood');
    legend('TMoE log-likelihood');
    legend('boxoff')
    box on;
else %eg. for real data with unknown classes etc
    
    [x, indx] = sort(x);
    y = y(indx);
    stats.Ey_k = stats.Ey_k(indx,:);
    stats.Piik = stats.Piik(indx,:);
    stats.Ey = stats.Ey(indx);
    stats.Vy = stats.Vy(indx);
    stats.klas = stats.klas(indx);
    
    figure,
    h1 = plot(x,y,'o','color',[0.6 0.6 .6]);
    hold all
    h2 =  plot(x,stats.Ey_k,'r--');
    h3 =  plot(x,stats.Ey,'r','linewidth',2);
    xlabel('x'), ylabel('y');
    ylim(yaxislim)
    hold off
    legend([h1, h2(1), h3], 'data','TMoE mean function','Estimated Experts',...
        'Location','SouthWest');
    legend('boxoff')
    figure,
    for k=1:K
        plot(x,stats.Piik(:,k),[color{k},'-'],'linewidth',2);
        hold on;
    end
    hold off
    ylim([0, 1]);
    xlabel('x'), ylabel('Gating network probabilities');
    
    
    %% data and Estimated mean functions and pointwise 2*sigma confidence regions
    figure,
    h1 = plot(x,y,'o','color',[0.6 0.6 .6]);
    hold all
    % estimated
    h2 = plot(x,stats.Ey,'r','linewidth',2);
    h3 = plot(x,[stats.Ey-2*sqrt(stats.Vy), stats.Ey+2*sqrt(stats.Vy)],'r--','linewidth',1);
    legend([h1, h2(1), h3(1)],'data', 'Estimated mean (TMoE)', 'Estimated conf. regions',...
        'Location','SouthWest');
    legend('boxoff');
    xlabel('x'), ylabel('y');
    ylim(yaxislim)
    hold off
    
    %% obtained partition
    figure
    hold all
    for k=1:K
        h1= plot(x,stats.Ey_k(:,k),color{k},'linewidth',1.2);
        h2= plot(x(stats.klas==k),y(stats.klas==k),[color{k},'o']);
        hold on
    end
    legend('Estimated expert means','Estimated clusters');
    ylim(yaxislim)
    box on
    xlabel('x'), ylabel('y');
    
    legend('boxoff')
    hold off
    %% observed data log-likelihood
    figure, plot(stats.stored_loglik,'-');
    xlabel('EM iteration number');
    ylabel('Observed data log-likelihood');
    legend('TMoE log-likelihood');
    legend('boxoff')
    box on;
end
end