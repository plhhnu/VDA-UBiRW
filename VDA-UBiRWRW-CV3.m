clear
clc
Sm_Vi_inl = xlsread(sm-vi);
vi_sim = xlsread(vi-sim);
sm_sim = xlsread(sm-sim;
[m,n]=size(Sm_Vi_inl);
[X1,Y1]=find(Sm_Vi_inl==1);
[X2,Y2]=find(Sm_Vi_inl==0);
[Len,Hig]=size(X2);
W_vi_sm_inl=Sm_Vi_inl;
t=0;
alpha=0.45;
Num=10;
l1=1;
l2=4;
W_sm_vi=Sm_Vi_inl;
SumAuc=0;
SumAcc=0;
SumSen=0;
SumSpe=0;
cn=0;
%================================================
for count=1:Num
 indices = crossvalind('Kfold',407,5);%正样本分割
for i =1:5
    Pos = (indices==i);
    PosDataX = X1(Pos,:);%正样本X
    PosDataY= Y1(Pos,:);%正样本Y
    [c,l]=size(PosDataX);
    P_sm_vi_inl=(1/(407))*Sm_Vi_inl;
   %===========================
    B=Sm_Vi_inl;
    for cross=1:c
        B(PosDataX(cross,1),PosDataY(cross,1))=0;
        P_sm_vi_inl(PosDataX(cross,1),PosDataY(cross,1))=0;
    end
    
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
Label=zeros(1,Len+c);%标签
Score=zeros(1,Len+c);%score
for cross=1:c
        Score(1,cross)=Wvism(PosDataX(cross,1),PosDataY(cross,1));
        Label(1,cross)=1;
end
 for cross=1:Len
        Score(1,cross+c)=Wvism(X2(cross,1),Y2(cross,1));
 end
%==================ROC曲线========================
predict = Score;
ground_truth = Label;
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
avescore=sum(Score)/(Len+c);
postive=Score(1:c);
negative=Score(c+1:Len+c);
TP=sum(postive>=avescore);
FP=c-TP;
TN=sum(negative<avescore);
FN=Len-TN;
Acc=(TP+TN)/(Len+c);
Sen=TP/c;
Spe=TN/Len;
SumSen=SumSen+Sen;
SumAcc=SumAcc+Acc;
SumSpe=SumSpe+Spe;
end   
end
AveAuc=SumAuc/(5*Num)
AveSen=SumSen/(5*Num)
AveAcc=SumAcc/(5*Num)
AveSpe=SumSpe/(5*Num)
%finalSM = xlswrite('/Users/liufx/SD/Article/2020/2019-CNOV/COVID-19/sm-vi-nov.xlsx',Wvism)
