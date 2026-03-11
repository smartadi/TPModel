function Uco=genhullall(U,types)

%execute command genhull for all dimension
%types of the hulls are gigven such type=["snnn" "irno" "box"]
%	possible values of type:
%		'eye': return eye(size(U{i},1))
%		'ortho': keep as is: W{i} = U{i} (not hull matrix!)
%		'snnn': general non-negative sum-normalized (convex combination)
%		'cno': 'snnn' and the values of each columns get close to 1 (tight hull)
%		'irno': 'snnn' and the values of each columns get close to 0
%		'box': perform box_decomp


for i=1:length(types)
   if types(i)~="none"
    Uco{i}=genhull(U{i},types(i));
   else 
    Uco{i}=[];
   end
end