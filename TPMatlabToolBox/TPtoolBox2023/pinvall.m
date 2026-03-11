function PU=pinvall(Uco)

% generate the pseudo inverse of all Uco (cell type)

for i = 1:length(Uco)
	if ~isempty(Uco{i})
		PU{i}=pinv(Uco{i});
    else PU{i}=[];
	end
end
