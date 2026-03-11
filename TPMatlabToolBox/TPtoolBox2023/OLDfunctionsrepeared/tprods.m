function T = tprods(S, U)
%Tensor product of a tensor and matrices 
%	
%	S  - Tensor - N dimensional
%	U  - Matrices (on cell type U{n})for each dimension of S
%	     if U{n}==[] or length(U)=n<N then no multiplication will be done
%	     in dimensions >n.
%	
%	T  - result of the product
%
%	eg. tprod(ones(2,3,4), {ones(5,2), [], ones(6,4)})
%
%	See also tprodn

T = S;
siz = size(S);
for i = 1:length(U)
	if ~isempty(U{i})
		siz(i) = size(U{i},1);
		H = layout(T, i);
		T = ndim_fold(U{i}*H, i, siz);
	end
end







