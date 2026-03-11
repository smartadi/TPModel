clear all
close all

f=@(p) [2+sin(p);
        2+cos(p)];

H=200;               % H = number of grid 
Omega=[0 2*pi];     % intervall of p
R=2;                % Number os singuar values to keep
g=linspace(Omega(1),Omega(2),H); % Gridding

for i=1:H  % sampling
    p=g(i); F(i,:)=f(p);
end

[U,D,V]=svd(F);
diag(D)
Uc=U(:,1:R);                % truncating the singuar values
Usnnn=genhull(Uc,'snnn');   % SNNN-tpye weighing functions
Ucno=genhull(Uc,'cno');     % CNO
Uirno=genhull(Uc,'irno');   % IRNO

Ssnnn=pinv(Usnnn)*F;        % get vertices        
Scno=pinv(Ucno)*F;          % get vertices
Sirno=pinv(Uirno)*F;        % get vertices

Esnnn=norm(F-Usnnn*Ssnnn)   % error over the grid
Ecno=norm(F-Ucno*Scno)
Eirno=norm(F-Uirno*Sirno)

figure('DefaultAxesFontSize',24)
plot(g,Usnnn,'LineWidth',4); grid on; title('SNNN');

figure('DefaultAxesFontSize',24)
plot(g,Ucno,'LineWidth',4); grid on; title('CNO');

figure('DefaultAxesFontSize',24)
plot(g,Uirno,'LineWidth',4); grid on; title('IRNO');

