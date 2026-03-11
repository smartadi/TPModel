function [Sco Uco sv S U]=ConvexTPStructure(F,hulltypes,tol,manual)

% Sco is the convex core tensor
% Uco (Cell type) are the 'snnn', 'irno', 'cno', 'eye' or 'box' type
% matrices
% sv (Cell type) are the singular values 
% S and U are the resuots of the HOSVD
% if manual is 1 then we can select the singular values to keep. if manual
% is 0 then it is done automatically accorind to tol.
%tol is a vector that defines the treshold to cut the smaller 
% singular values. 


for i=1:length(hulltypes)
    dim(i)=1;
    if hulltypes=="none"
        dim(i)=0;
    end 
end

[S,U,sv]=hosvd(F,dim,tol);

if manual==1
    for i=1:length(hulltypes)
        dimension=i;
       if hulltypes(i)~="none"
            dimension
            display('Singular values')
            sv{i}
            num=input('singular values to keep?')
            U{i}=U{i}(:,1:num);
            Uco{i}=genhull(U{i},hulltypes(i));
        else
            Uco{i}=[];
        end
    end
else
    Uco=genhullall(U,hulltypes);
end

PUco=pinvall(Uco);
Sco=tprod(F,PUco);
PU=pinvall(U);
S=tprod(F,PU);