function X = designmatrix_Poly_Reg(x,p)
%
%
%
%
%
%
%
%
%
%
%
%
%
%
% by Faicel Chamroukhi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (size(x,2) ~= 1)
    x=x'; % a column vector
end

n = length(x);
X=zeros(n,p+1);
for i = 1:p+1
    X(:,i) = x.^(i-1);% [1 x x.^2 x.^3 x.^p;......;...]
end