% RNOINO - mátrix felbontás az RNO és INO feltételek szerint
%
% [IP,BASE]=RNOINO(U)
% 
% Az U mátrix nxr-es, n>r és SN, azaz a sorok összege
% 1 kell legyen;
% 
% A kimenet az nxr-es IP és az rxr-es BASE mátrix, ahol
% a BASE SN, 
% az IP pedig SN  (sorok összege 1), 
%             NN  (nemnegatív elemu) 
%             INO (minden oszlop legkisebb eleme 0)
%          és RNO (minden oszlop legnagyobb eleme azonos,0 és 1 közötti)
% 
% továbbá U=IP*BASE
% Corrected by:  Patricia Grof, 2012, ppgrof@mae.cuhk.edu.hk

function [U,fi] = rnoino(U0)

U=U0;
[n,r]=size(U0);

i=r-1;

vet(:,i)=U(:,i+1);%a pontok tavolsaga a levetitettektol
U=U(:,1:i);
U=U+vet(:,i)*ones(1,i)/i; %vetites az x1+..+x(i-1)=1 sikr
    
U=U-vet(:,i)*ones(1,i)/i;
U=[U 1-sum(U,2)];
a=[0 1];
while rnodiff(U,a(2))<0  %a megoldas keresesi tartomanya
    a(2)=a(2)*2;
end
d=[rnodiff(U,a(1)),rnodiff(U,a(2))];
while abs(d(1))+abs(d(2)) > 10^(-6) %kereses oroszlanfogassal(intervallumfelezes)
    auj=sum(a)/2;
    if rnodiff(U,auj) > 0
        a(2)=auj;
    else
        a(1)=auj;
    end
    d=[rnodiff(U,a(1)),rnodiff(U,a(2))];
end     %a vegen auj adja a megfelelo parametererteket

auj=sum(a)/2;
U00=U(:,1:(size(U,2)-1));
s=sum(U00,2);
U1=U00+(auj-1)/i*s*ones(1,i);
U2=U1-ones(n,1)*min(U1);
U25=ones(n,1)*max(U2);
U3=U2./U25;
U4=U3./max(sum(U3,2)');
U=[U4 1-sum(U4,2)];

fi=inv(U(1:r,:))*U0(1:r,:);