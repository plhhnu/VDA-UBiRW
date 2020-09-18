clear
clc
Sm_Vi_inl = xlsread(sm-vi);
vi_sim = xlsread(vi-sim);
sm_sim = xlsread(sm-sim;
[m,n]=size(Sm_Vi_inl);
W_vi_sm_inl=Sm_Vi_inl;
t=0;
l1=1;
l2=4;
alpha=0.45;
Num=100;
W_sm_vi=Sm_Vi_inl;
SumAuc=0;
SumAcc=0;
SumSen=0;
SumSpe=0;
CNum=100000;
cn=0;
%================================================
for count=1:Num
 indices = crossvalind('Kfold',n,5);%正样本分割
 colind=indices';
for i =1:5
    Pos = (colind==i);
    c=sum(sum(Sm_Vi_inl(:,Pos)));
    P_sm_vi_inl=(1/(407))*Sm_Vi_inl;
   %===========================
    P_sm_vi_inl(:,Pos)=0;
    B=Sm_Vi_inl;
    B(:,Pos)=0;
  %==========================PRW===============================
   vi_index=(sum(B)==0);
   sm_index=(sum(B,2)==0);
   
 vi_num=(1./(sum(B)));
 vi_num(vi_index)=0;
 
sm_num=(1./(sum(B,2)));
sm_num(sm_index)=0;

vi_m=repmat(vi_num,m,1);
sm_n=repmat(sm_num,1,n);


P1_sm_vi=(sm_sim)*diag(sm_num);
P1_sm=repmat(1./sum(P1_sm_vi,2),1,m);
P2_sm_vi=vi_sim*diag(vi_num);
P2_vi=repmat(sum(1./P2_sm_vi,1),n,1);
infindex= (P2_vi==Inf);
P2_vi(infindex)=1/n;
M=max(l1,l2);
n1=0;
n2=0;
while (cn<=M)
    if (t==0)
        
        if(n1<=l1)
         W1=alpha*((sm_sim)*P_sm_vi_inl)+(1-alpha)*P_sm_vi_inl;
         n1=n1+1;
        end
        
        if(n2<=l2)
        W2=alpha*(P_sm_vi_inl*(vi_sim))+(1-alpha)*P_sm_vi_inl;
        n2=n2+1;
        end
         W_sm_vi=(W1+W2)/2;
         t=t+1;
    end
    if(t~=0)
        W_sm_vi=1/sum(sum(W_sm_vi))*W_sm_vi;

        if(n1<=l1)
        W1=alpha*((sm_sim)*W_sm_vi)+(1-alpha)*P_sm_vi_inl;
         n1=n1+1;
        end
        
        if(n2<=l2)
        W2=alpha*(W_sm_vi*(vi_sim))+(1-alpha)*P_sm_vi_inl;
        n2=n2+1;
        end
        W_sm_vi=(W1+W2)/2;
        
    end
    cn=cn+1;
    
 end


Wvism=W_sm_vi;

%==================Label and Score=================
Label=Sm_Vi_inl(:,Pos);%标签
Score=Wvism(:,Pos);%score
[ml,nl]=size(Label);
[ms,ns]=size(Score);
%==================ROC曲线========================
predict =reshape( Score,1,ms*ns);
ground_truth = reshape( Label,1,ml*nl);
x= 1.0;
y = 1.0;
pos_num = sum(ground_truth==1);
neg_num = sum(ground_truth==0);
x_step=1.0/neg_num;
y_step=1.0/pos_num;
mn = length(ground_truth);
[predict,index] = sort(predict);
ground_truth = ground_truth(index);
for i = 1:mn
   if  ground_truth(i)==1
       y=y-y_step;
   else
       x=x-x_step;
   end
X(i)=x;
Y(i)=y;
end
auc= -trapz(X,Y);
plot(X,Y,'-ro','LineWidth',2,'MarkerSize',3);
xlabel('FPR');
ylabel('TPR');
title(['ROC curve of  (AUC=' num2str(auc) ')']);
SumAuc=SumAuc+auc;
%===========================================

avescore=sum(predict)/(ms*ns);
Positive=predict(ground_truth==1);
Negative=predict(ground_truth==0);
TP=sum(Positive>=avescore);
FP=sum(ground_truth==1)-TP;
TN=sum(Negative<avescore);
FN=sum(ground_truth==0)-TN;
Acc=(TP+TN)/(sum(ground_truth==0)+sum(ground_truth==1));
Sen=TP/sum(ground_truth==1);
Spe=TN/sum(ground_truth==0);
SumSen=SumSen+Sen;
SumAcc=SumAcc+Acc;
SumSpe=SumSpe+Spe;
end   
end
AveAuc=SumAuc/(5*Num)
AveSen=SumSen/(5*Num)
AveAcc=SumAcc/(5*Num)
AveSpe=SumSpe/(5*Num)
%finalSM = xlswrite('/Users/liufx/SD/Article/2020/2019-CNOV/COVID-19/sm-vi.xlsx',Wvism)
