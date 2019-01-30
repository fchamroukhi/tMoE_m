function x = log_normalize(x)
% compute x - repmat(log(sum(exp(x),2)),1,size(x,2)) in a robust way
[n d] = size(x);
a = max(x,[],2);
x = x - repmat(a + log( sum( exp( x - repmat(a,1,d) ) , 2 ) ) ,1,d) ;
