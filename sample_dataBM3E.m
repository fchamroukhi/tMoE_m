
function [x, r]= dataBM3E(n)
% Jeux de données BM3E Discriminative density propagation for visual
% tracking

%N=250;
x = rand(n,1);
r = x + 0.3 *sin(2*pi*x) + normrnd(0,0.05,n,1);
% plot(r,x,'bo'), xlim([0 1.5]),ylim([0 1.4])
% xlabel('Input r'),ylabel('Output x')
