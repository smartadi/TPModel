function M = layout(T, dim)
%Layout an array in a given dimension to a matrix
%	T   - tensor
%	M   - matrix containing all vectors of dimension "dim" of tensor T.
%	See also NDIM_FOLD.

ndim = ndims(T);
if ndim == 2
	% T is 2D
	if dim == 2
		M = T';
	else
		M = T;
	end
else
	% Rotate tensor T	
	dimorder = 1:ndim;
	newdimorder = wshift('1D', dimorder, dim-1);
	T = permute(T, newdimorder);
	
	% layout the rotated T
	siz = size(T, 1);
	M = reshape(T, siz, []);
end
