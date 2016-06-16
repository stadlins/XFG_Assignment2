


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%  1.  de-GARCH for stock return       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%   2.  Using residuals to estimate dependence structure from candidate copula functions   %%%%%%%%%%%%
%%%%%%%%%%%%%   3.  simulate d-dimentional dependence based on  copula               %%%%%%%%%%%%
%%%%%%%%%%%%%   4.  Portfolio Value-at-Risk estimation               %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc
clear all
tic

data=xlsread('assignmet2data.xlsx','Tabelle1','B:K');  % stock return daily data
Size=xlsread('assignmet2data.xlsx','Tabelle2','C:C');                % firm size

[T,N]=size(data);
Return=log(data(2:T,:)./data(1:(T-1),:)); % calculate stock return
Return=Return*100;                        % Panel returns series across N SIFI

% Using first 1,5 years
Y=392; 

h=385;  % backward period for estimating GARCH residuals
Temp=Return((Y-h):(Y-1),:);
DeMeanReturn=Temp-repmat(mean(Temp),h,1);

Par=zeros(N,3);   % p,o,q  where marginal is normal cdf
% Par=zeros(N,4);   % p,o,q  where marginal is t cdf
Ht=zeros(h,N);

% GARCH estimation

for i=1:N     
    [Par(i,:), LL, Ht(:,i)]=tarch(DeMeanReturn(:,i),1,0,1);    % error distribution as default "Normal"
%      [Par(i,:), LL, Ht(:,i)]=tarch(DeMeanReturn(:,i),1,0,1,'STUDENTST'); 
end

% DeGARCH effect
Residuals=DeMeanReturn./sqrt(Ht);

%%%%  Estimate dependence parameters in copula density   %%%% 

theta = zeros(N,N);  % Gaussian dependence 
theta_t = zeros(N,N);
theta_Clay = zeros(N,N);
theta_Gum = zeros(N,N);
tau_Clay = zeros(N,N);
tau_Gum = zeros(N,N);
tail_Upper = zeros(N,N);
tail_Lower = zeros(N,N);

nu = zeros(N,N);


 for i=1:N
       V=normcdf(Residuals(:,i));     
%         V=tcdf(Residuals(:,i));  
        for j=(i+1):N    
            U=[normcdf(Residuals(:,j)), V];
%             U=[tcdf(Residuals(:,j)), V];
            K=  copulafit('Gaussian',U);
            [ theta(j,i) ]=K(1,2);
            [K, P]=  copulafit('t',U);
            [ theta_t(j,i) ]=K(1,2);
            nu(j,i)=P;
            [ theta_Clay(j,i) ] =  copulafit('Clayton',U);
            [ theta_Gum(j,i) ] =  copulafit('Gumbel',U);
            
             tau_Clay(j,i)    =   theta_Clay(j,i)/( theta_Clay(j,i)+2);   %relate Clayton dependence to kendall tau
             tau_Gum(j,i)     =   1-1/theta_Gum(j,i);                     %relate Gumbel dependence to kendall tau
             tail_Upper(j,i)  =   2-2^(1/theta_Gum(j,i));                 % uppper tail dependence from Gumbel
             tail_Lower(j,i)  =   2^(1/theta_Clay(j,i));                  % Lower tail dependence from Clayton
             
             %Symmetric dependence between i and j
             theta(i,j)=theta(j,i);
             theta_t(i,j)=theta_t(j,i);
             theta_Clay(i,j)=theta_Clay(j,i);
             theta_Gum(i,j)=theta_Gum(j,i);
             tau_Clay(i,j)=tau_Clay(j,i);
             tau_Gum(i,j)=tau_Gum(j,i);
             tail_Upper(i,j)=tail_Upper(j,i);
             tail_Lower(i,j)= tail_Lower(j,i);
                          
        end
             theta(i,i)=1;
             theta_t(i,i)=1;
             tau_Clay(i,i)=1;
             tau_Gum(i,i)=1;
             tail_Upper(i,i)=1;
             tail_Lower(i,i)= 1;
             theta_Clay(i,i)=999;
             theta_Gum(i,i)=999;
       
 end

% % % %%%%%   Gererate d-dimentional copula r.v.   based on copula dependence matrix%%%

G_U=zeros(1000,N,2);
G_RV=zeros(1000,N,2);
nu=5;

% value-weighted noncentral nodes
   
Portfolio=zeros(1000,N);
PVaR=zeros(2,2);


%%%%   Copula simulation in d dimension   %%%%%

G_U(:,:,1) =copularnd('Gaussian', theta,1000);   %  dimension  1000*(d-1)      
G_RV(:,:,1) = norminv(G_U(:,:,1) ,0,1) +repmat(mean(Temp),1000,1);
G_U(:,:,2) =copularnd('t', theta_t,nu,1000);   %  dimension  1000*(d-1)      
G_RV(:,:,2) = norminv(G_U(:,:,2) ,0,1) +repmat(mean(Temp),1000,1);

Weight=Size./sum(Size); % value-weighted  


for p=1:2     % Two types of copula (Gassian and t)
    
    Portfolio(:,p)=G_RV(:,:,p)*Weight;
    PVaR(p,:)=quantile(Portfolio(:,p),[0.01,0.05]);
   
end


