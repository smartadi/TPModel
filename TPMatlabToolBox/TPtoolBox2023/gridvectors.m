function G=gridvectors(Omega,density)

% G is a Cell type. Each element of the Cell is a vector defining a grid.
% Omega is a matrix the rows are assigned to the dimensions. each row 
% defines the minimum and the maximum value of the related diemnsion.
% density is a vector. Each element defines the number of grid on the
% corresponding dimension. 

for n=1:length(density)
    G{n}(:)=linspace(Omega(n,1),Omega(n,2),density(n));
end
