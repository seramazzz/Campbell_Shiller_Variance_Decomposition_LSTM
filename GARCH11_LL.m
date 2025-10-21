function [LL,ht,zt]=GARCH11_LL(parm,e_t)
%This function calculates the log-likelihood of a GARCH11
%model under normal assumption
%parm is a 3x1 vector collecting all parmeters of a
%GARCH11 model
%e_t is a Tx1 vector of residuals
%LL returns the likelihood of the model
%ht returns the conditional variance estimates given parm
%zt returns the standardized innovation

%This is the order of the parameters in parm
%mu=parm(1);
omega=parm(1);
alpha=parm(2);
beta=parm(3);

%e_t=data-mu;
h0=var(e_t);%initial value for ht
T=length(e_t);%Get the dimension of the data
ht=zeros(T,1);%pre-allocate a vector for ht
zt=zeros(T,1);%pre-allocate a vector zt
ht(1)=omega+beta*h0;%set h1
LL=0; %pre-allocate log-likelihood
    for i=2:T %a loop to construct GARCH(1,1) iteratively
        ht(i)=omega+alpha*e_t(i-1)^2+beta*ht(i-1);
        zt(i)=e_t(i)/sqrt(ht(i));
        LL=LL-0.5*log(2*pi)-0.5*log(ht(i))-0.5*zt(i)^2;
    end
end