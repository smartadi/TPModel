function T = ndim_fold(M, dim, siz)
%   Restore a matrix into a tensor with the given size
%	M   - matrix
%	dim - the columns of M will turn into this dimension of T
%	siz - size(T) (prod(siz)==length(M), siz(dim)==size(M,1) should hold)

%	See also layout.

ndim = length(siz);

if ndim == 2
	% T is 1D or 2D
	if dim == 2
		T = M';
	else
		T = M;
	end
else
	
	% restore M into T
	new_size = wshift('1D', siz, dim-1);
	new_size(1) = size(M, 1);
	T = reshape(M, new_size);
	
	% rotate T into the right position
	dim_order = wshift('1D', [1:length(siz)], -(dim-1));
	T = permute(T, dim_order);
end
