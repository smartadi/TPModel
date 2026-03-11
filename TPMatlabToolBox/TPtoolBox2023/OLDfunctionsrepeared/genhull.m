function [W V] = genhull(U, wtype)
% Generate "hull" matrices from orthonormal matrices
%	W = GENHULL(U)
%	W = GENHULL(U, wtype)
%	[W V] = GENHULL(U)
%	[W V] = GENHULL(U, wtype)
%	
%	U     - matrices in a cell with orthonormal column vectors (U{i}' * U{i} = I)
%	wtype - weight function type for each matrix (default: 'snnn')
%	
%	W     - matrices controlling the convex hull
%	V     - vertex matrices (W{i}*V{i} = U{i})
%	
%	possible values of wtype:
%		'eye': return eye(size(U{i},1))
%		'ortho': keep as is: W{i} = U{i} (not hull matrix!)
%		'snnn': general non-negative sum-normalized (convex combination)
%		'cno': 'snnn' and the values of each columns get close to 1 (tight hull)
%		'irno': 'snnn' and the values of each columns get close to 0
%		'box': perform box_decomp
%	
%	See also HOSVD, DECOMP.

if nargin <= 1
	% TODO
	wtype = 'close';
end


if ~iscell(U)
	% only one matrix
	if nargout < 2
		W = decomp(U, wtype);
	else
		[W V] = decomp(U, wtype);
	end
	return
end

if ~iscell(wtype)
	tmp = wtype;
	wtype = cell(1,length(U));
	for i = 1:length(U)
		wtype{i} = tmp;
	end
end


% TODO: allow skipping last few U?
%assert(length(U) == length(wtype));

W = cell(1,length(wtype));
if nargout < 2
	for i = 1:length(wtype)
		if isempty(U{i})
			W{i} = [];
		else
			W{i} = decomp(U{i}, wtype{i});
		end
	end
else
	V = cell(1,length(wtype));
	for i = 1:length(wtype)
		if isempty(U{i})
			W{i} = [];
			V{i} = [];
		else
			[W{i} V{i}] = decomp(U{i}, wtype{i});
		end
	end
end
