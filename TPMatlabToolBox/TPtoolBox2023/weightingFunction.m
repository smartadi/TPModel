function w=weightingFunction(p,U,G)

% Returns the values of the weighting functions for each dimension over p.
% p is a vector
% w (cell type) is the vectors containing the values of the weighting functions over p
% ineach diemnsion
% U (cell) contains the matrices for each dimension. These define the weighting functions. 
% G is a Cell containing the grid vectors for each dimension. 


w = cell(1,length(U));
for n = 1:length(U)
    if ~isempty(U{n})
        if p(n) < G{n}(1)
            w{n} = U{n}(1,:);
	    elseif p(n) > G{n}(end)
		    w{n} = U{n}(end,:);
	    else
		    w{n} = interp1(G{n}, U{n}, p(n));
        end
    else w{n}=[];
    end
end
