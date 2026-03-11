function y=TPmodel(p,S,U,G)

% p is th evector of the inputs
% S core tensor of the vertices
% U (Cell) matrix of the weighting functiöns
% G (Cell) grid vectors
N=length(G);
SS=size(S);
blockdim=length(SS)-N;
w=weightingFunction(p,U,G);
ygross=tprod(S,w);

if blockdim==0
    y=ygross;
end
 if blockdim==1;
    y=shiftdim(ygross,1);
else
    y=reshape(ygross,SS(N+1:end));
end


%if length(size(S))==N
%    y=ygross(1,1);
%end
%if length(size(S))==N+1
%    y(:)=ygross(1,1,1,:);
%end
%if  length(size(S))>N+1
%    shape=size(S);
%    y=reshape(ygross,shape(N+1:end));
%end
