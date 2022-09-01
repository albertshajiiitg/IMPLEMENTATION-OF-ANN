%% artificial neural network implementation
clc,clear all;
load data_set.txt;
ann_data= data_set; 
data=ann_data(:,[1:4,6]);
P=40;                     %% number of training patterns
Pt=3;                     %% number of test patterns
L = 4;                    %% number of inputs considered is 4
N = 1;                    %% number of outputs is 1         
M = 9;                    %% number of hidden neurons
eta=0.1;                  %% learning rate
mt= 0.75;                   %% momentum term
% get target values of test dataset
target_test=data(P+1:end,end)';
% get max and min values of target for training dataset
max_target = max(data(1:P,end));
min_target = min(data(1:P,end));
maximum = max(data,[],1);                            
minimum = min(data,[],1);
[rows , cols ] = size(data);               %% normalizing input and output values 
for i=1:rows
    for j=1:cols
        data(i,j) = 0.1 + 0.8*((data(i,j) - minimum(j))/(maximum(j)-minimum(j)));
    end
end
data1=data(1:P,:);
data1= data1(randperm(size(data1, 1)), :);
I = ones(L+1,P);
I_test= ones(L+1,Pt);
I(2:L+1,1:P)=data1(1:P,1:L)';
T=data1(1:P,end)';
I_test(2:L+1,1:Pt)=data(P+1:end,1:L)';
T_test=data(P+1:end,end)';
v=ones(L+1,M);      %% initializing the connection weight values including the bias 
w=ones(M+1,N);
for l=2:L+1                                             
    for m=1:M
        v(l,m) = random('Normal',0,0.333); %% normalizing the connection weight matrices
    end
end

for m=2:M+1
    for n=1:N
        w(m,n) = random('Normal',0,0.333);
    end
end
itrs = 0;
MSE =1;
pre_delta_v = zeros(L+1,M); 
pre_delta_w = zeros(M+1,N);  
while itrs< 1000000        %% training starts 
    IH=zeros(M,P);
    OH=zeros(M,P);
    IO=zeros(N,P);
    OO=zeros(N,P);
    for p=1:P
        for m=1:M
            for l=1:L+1
                IH(m,p)=IH(m,p)+I(l,p)*v(l,m);
            end
        end
    end
    OH = 1 + exp(-1*IH);
    OH=1./OH;
    OH = [ones(1,P);OH];
    for p=1:P
        for n=1:N
            for m=1:M+1
                IO(n,p) = IO(n,p) + OH(m,p)*w(m,n);
            end
        end
    end
    OO = 1 + exp(-1*IO);
    OO=1./OO;
    E1 = ((T - OO).^2);
    E2=0.5*E1;
    E2=sum(E2,2);
    MSE=sum(E2)/P;
    delta_v=zeros(L+1,M);
    delta_w=zeros(M+1,N);
    for p=1:P
        for m=1:M+1
            for n=1:N
                delta_w(m,n)=delta_w(m,n)+(T(n,p)-OO(n,p))*OO(n,p)*(1-OO(n,p))*OH(m,p);                
            end
        end
    end
    delta_w=delta_w*(eta/P);
    for p=1:P
       for l=1:L+1
           for m=1:M
               for n=1:N     
                   delta_v(l,m)=delta_v(l,m)+(T(n,p)-OO(n,p))*OO(n,p)*(1-OO(n,p))*w(m+1,n)*OH(m+1,p)*(1-OH(m+1,p))*I(l,p);
               end
           end
       end
    end
    delta_v=delta_v*(eta/(N*P));
    w=w+delta_w +mt*pre_delta_w;   
    v=v+delta_v +mt*pre_delta_v;   
    pre_delta_v=delta_v;  
    pre_delta_w=delta_w;    
    itrs=itrs+1;
    MSE_values(itrs)=MSE;
end
OO = zeros(N,Pt);            %% calculation of output values for test set starts 
IH = zeros(M,Pt);
OH = zeros(M,Pt);
IO = zeros(N,Pt);
for p=1:Pt
    for m=1:M 
        for l=1:L+1
             IH(m,p) = IH(m,p)+ I_test(l,p)*v(l,m);
        end
     end
end
OH = 1 + exp(-1*IH);
OH = 1./OH;
OH = [ones(1,Pt);OH];
for p=1:Pt
    for n=1:N 
        for m=1:M+1
            IO(n,p) = IO(n,p) + OH(m,p)*w(m,n);
        end
    end
end
OO = 1 + exp(-1*IO);
OO = 1./OO;
E1 = (T_test - OO).^2;
E2 = 0.5*E1;
E2 = sum(E2,2);
MSE_test = sum(E2)/Pt;
fi = fopen('output_data.txt','w');  %% output is printed in out_data txt file
fprintf(fi,'total iterations during training %d\n',itrs);
fprintf(fi,'MEAN SQUARE ERROR for the training set :%f\n',MSE);
fprintf(fi,'MEAN SQUARE ERROR for the test set :%f\n',MSE_test);
fprintf(fi,'\nThe output of the test dataset is \n');
for p=1:Pt
    for n=1:N
         fprintf(fi,'%d\n',target_test(n,p));
     end
end    
fprintf(fi,'\nThe output of the Neural network is \n');
for p=1:Pt
     for n=1:N
        OO(n,p) = ((OO(n,p)-0.1)*((max_target-min_target)/0.8)) + min_target;
     end
end

for p=1:Pt
    for n=1:N
         fprintf(fi,'%d\n',OO(n,p));
     end
end
fprintf(fi,'\nThe error in prediction is \n');
for p=1:Pt
    for n=1:N
         fprintf(fi,'%d\n',abs(OO(n,p)-target_test(n,p)));
     end
end

%{
plot([1:itrs],MSE_values)              %% plotting mse during training      
legend("MSE");
xlabel("Number of iterations");
ylabel("MSE");
title("MEAN SQUARE ERROR VS  ITERATIONS");
%}


