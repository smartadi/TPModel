clear

%THIS WORKS WITH TPtoolBox2023
% THE NEW FUNCTIONS AND REPEARED OLDFUNCTIONS

% PLEASE DO NOT FORGET TO SET THE PATH 
% in your MATLAB to the subfolders of the 
% folder TPtoolBox2023.

% f(p) where p in Omega. In this example p in R^3.
% Goal is to find f(p) = Sco \times_{n=1}^N Uco_n(p_n)

% STEP 1: set Omega and the grid 
Omega=[1 10; % interval of p1
       2 20; % interval of p2
       3 30]; % interval of p3
G=[30 40 50]; % number of the grid on p1,p2 and p3 respectively
DG=gridvectors(Omega,G); 
%DG is a cell type, each DG{n} is a vector that defines the values of the 
% grid on p_n

% the next step is to sample f(p) over the grid. 
% Let f(p) in R^{2\times 3}. It is defined at the end of this file

for i1=1:G(1)
    for i2=1:G(2)
        for i3=1:G(3)
            p1=DG{1}(i1);
            p2=DG{2}(i2);
            p3=DG{3}(i3);
            FD(i1,i2,i3,:,:)=fexample(p1,p2,p3);
        end
    end
end

% STEP 2
% we execute the HOSVD and then the either SNNN, CNO or IRNO 
% transfromation. Note that CNO leads to a tight hull of the vertices. The
% IRNO as well. SNNN leads to a loose hull.
% since the parameter vector p is 3 diemsnional let us define 
% snnn, and cno and irno weighting function on those dimensions

[Sco,Uco,sv,S,U]=ConvexTPStructure(FD,["snnn" "cno" "irno"],[1e-9 1e-10 1e-8],0);

% Sco contains the vertices accoring to the type, "co" means convex 
% Uco (cell type) contains the points of the 
% weighting functions. Each column defines a
% weighting functoin
% sv is cell type defines the singular values. sv{n} are the singular 
% values on dimensio n.
% let the treshold to automatically cut the small singular values be 
% [1e-9]
% the last value is 0 that means the singular values are discarded
% automaticall according to [1e-9 1e-10 1e-8]. If it is 1 then we can do it
% manually. You may try.
% S and U are the direct result of the HOSVD without any convex
% transformation

%STEP 3

% the weighting membership functions are
figure(1)
plot(DG{1},Uco{1})
figure(2)
plot(DG{2},Uco{2})
figure(3)
plot(DG{3},Uco{3})

% you may see that two membership functions are defined by dimensions only


%the consecquents are - for instance

Vertex111(:,:)=Sco(1,1,1,:,:)
Vertex112(:,:)=Sco(1,1,2,:,:)


% the TS Fuzzy model is ready to use:

% the TPmodel of TS Fuzzy model can be calculated for any 
% p now as for instance

p=[1 5 9];

y=TPmodel(p,Sco,Uco,DG);



% the function to be transformed

function y=fexample(g1,g2,g3)
       y(:,:)=[g1^2+3*g2+1/g3 0 1; 1 0 g1^2+3*g2+1/g3];
end

