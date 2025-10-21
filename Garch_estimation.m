%% Directory
clearvars;
clc;
cd '/Users/seramaz1/Desktop/The University of Manchester/PhD/Machine Learning/Coursework'
%% A bit of data
% S&P 500
SP500 = readtable("SP500.xlsx");
SP500.Day = day(SP500.Date);
SP500.Month = month(SP500.Date);
SP500.Year = year(SP500.Date);
RF = readtable("RF.xlsx");
RF.Date=[];
%SP500 = join(SP500,RF,"LeftVariables",{'Year','Month','Day'});
rf = zeros(length(SP500.SP500),1);
for i = 1:length(SP500.SP500)
    if isempty(RF.Rf(RF.Year==SP500.Year(i) & RF.Day==SP500.Day(i) & RF.Month == SP500.Month(i))) == 1
        rf(i) = nan(1,1);
    else
        rf(i) = RF.Rf(RF.Year==SP500.Year(i) & RF.Day==SP500.Day(i) & RF.Month == SP500.Month(i));
    end
end
SP500.RF = rf;

%% Little bit more data

hf_data = readtable("5min Stock Market Indices.csv");
hf_data = removevars(hf_data,{'Domain', 'Type', 'CloseAsk', 'CloseBid', 'CloseMidPrice'});
hf_data.Date_Time = datetime(...
    extractBetween(string(hf_data.Date_Time),1,10) ...
    + ' ' ...
    + extractBetween(string(hf_data.Date_Time),12,19));
% Some data errors
hf_data.Last(hf_data.Date_Time == '31-Mar-2004 15:05:00' & hf_data.x_RIC == ".SPX", :) = hf_data.Last(hf_data.Date_Time == '31-Mar-2004 15:00:00' & hf_data.x_RIC == ".SPX", :);
hf_data.Last(hf_data.Date_Time == '21-Dec-2004 15:50:00' & hf_data.x_RIC == ".SPX", :) = hf_data.Last(hf_data.Date_Time == '21-Dec-2004 15:45:00' & hf_data.x_RIC == ".SPX", :);
hf_data.Last(hf_data.Date_Time == '21-Dec-2004 15:55:00' & hf_data.x_RIC == ".SPX", :) = hf_data.Last(hf_data.Date_Time == '22-Dec-2004 08:25:00' & hf_data.x_RIC == ".SPX", :);
hf_data.Last(hf_data.Date_Time == '11-Jan-2010 15:10:00' & hf_data.x_RIC == ".SPX", :) = hf_data.Last(hf_data.Date_Time == '11-Jan-2010 15:20:00' & hf_data.x_RIC == ".SPX", :);
hf_data.Last(hf_data.Date_Time == '11-Jan-2010 15:15:00' & hf_data.x_RIC == ".SPX", :) = hf_data.Last(hf_data.Date_Time == '11-Jan-2010 15:20:00' & hf_data.x_RIC == ".SPX", :);
% Carry on
hf_data.Date_Time = datetime(extractBetween(string(hf_data.Date_Time),1,11));
hf_data = hf_data(hf_data.x_RIC == ".SPX", :);
hf_data = removevars(hf_data,{'x_RIC'});
hf_data = hf_data(year(hf_data.Date_Time) >= 2000, :);
hf_data.LogRet = [nan; log(hf_data.Last(2:end)./hf_data.Last(1:end-1))];

%% Realised Variance

RV = zeros(length(unique(hf_data.Date_Time)),1);
dates = unique(hf_data.Date_Time);
for i = 1:length(unique(hf_data.Date_Time))
    RV(i) = sum(hf_data.LogRet(hf_data.Date_Time == dates(i)).^2, "omitnan");
end
RV = table(dates,RV);
RV.Properties.VariableNames = ["Date","RV"];
RV_temp = zeros(length(SP500.SP500),1);
for i = 1:length(SP500.SP500)
    if isempty(RV.RV(year(RV.Date)==SP500.Year(i) & day(RV.Date)==SP500.Day(i) & month(RV.Date)==SP500.Month(i))) == 1
        RV_temp(i) = nan(1,1);
    else
        RV_temp(i) = RV.RV(year(RV.Date)==SP500.Year(i) & day(RV.Date)==SP500.Day(i) & month(RV.Date)==SP500.Month(i));
    end
end
SP500.RV = RV_temp;

RV = table(dates,RV);
RV.Properties.VariableNames = ["Date","RV"];

%% Returns

SP500.Ret = [nan; SP500.SP500(2:end)./SP500.SP500(1:end-1) - 1];
SP500.LogRet = [nan; log(SP500.SP500(2:end)./SP500.SP500(1:end-1))];
figure
subplot(2,3,1)
plot(SP500.Date,SP500.Ret)
title("S&P 500 Simple Return")
subplot(2,3,2)
autocorr(SP500.Ret,10)
subplot(2,3,3)
parcorr(SP500.Ret,10)
subplot(2,3,4)
plot(SP500.Date,SP500.LogRet)
title("S&P 500 Log Return")
subplot(2,3,5)
autocorr(SP500.LogRet,10)
subplot(2,3,6)
parcorr(SP500.LogRet,10)

%% Squared Return
figure
subplot(2,3,1)
plot(SP500.Date,SP500.Ret.^2)
title("Simple S&P 500 Return Squared")
subplot(2,3,2)
autocorr(SP500.Ret.^2,30)
subplot(2,3,3)
parcorr(SP500.Ret.^2,30)
subplot(2,3,4)
plot(SP500.Date,SP500.LogRet.^2)
title("Log S&P 500 Return Squared")
subplot(2,3,5)
autocorr(SP500.LogRet.^2,30)
subplot(2,3,6)
parcorr(SP500.LogRet.^2,30)

%% Fit ARMA

pmax=5; 
qmax=5;
% a bit long here
%estimate from ARMA(0,0) to ARMA(20,20)
LOGL = zeros(pmax+1,qmax+1); %collecting maximized log-likelihood ie adding slot for zero (constant)
PQ = zeros(pmax+1,qmax+1); %number of parameters in the model
for p = 0:pmax
    for q = 0:qmax
        Mdl = arima(p,0,q);
        [~,~,logL] = estimate(Mdl,SP500.LogRet,'Display','off');
        LOGL(p+1,q+1) = logL;
        PQ(p+1,q+1) = p + q ; %the number of parameters (ignoring the constant)
    end
end

%% Let's choose optimal lags

LOGL = reshape(LOGL,(pmax+1)*(qmax+1),1);
PQ = reshape(PQ,(pmax+1)*(qmax+1),1);
[aic,bic,ic] = aicbic(LOGL,PQ+1,length(SP500.LogRet));
aicmat=reshape(aic,pmax+1,qmax+1); % Akaike's (1974) AIC
bicmat=reshape(bic,pmax+1,qmax+1); % Schwartz's (1979) BIC
aiccmat=reshape(ic.aicc,pmax+1,qmax+1); % Corrected AIC (AICc)
caicmat=reshape(ic.caic,pmax+1,qmax+1); % Consistent AIC (CAIC)
hqcmat=reshape(ic.hqc,pmax+1,qmax+1); % Hannan-Quinn's (1979) HQC
%find the optimal ar and ma lags
[ar_opt_aic,ma_opt_aic]=find(aicmat==min(aicmat,[],'all'));
ar_opt_aic = ar_opt_aic - 1;
ma_opt_aic = ma_opt_aic - 1;
[ar_opt_bic,ma_opt_bic]=find(bicmat==min(bicmat,[],'all'));
ar_opt_bic = ar_opt_bic - 1;
ma_opt_bic = ma_opt_bic - 1;
[ar_opt_aicc,ma_opt_aicc]=find(aiccmat==min(aiccmat,[],'all'));
ar_opt_aicc = ar_opt_aicc - 1;
ma_opt_aicc = ma_opt_aicc - 1;
[ar_opt_caic,ma_opt_caic]=find(caicmat==min(caicmat,[],'all'));
ar_opt_caic = ar_opt_caic - 1;
ma_opt_caic = ma_opt_caic - 1;
[ar_opt_hqc,ma_opt_hqc]=find(hqcmat==min(hqcmat,[],'all'));
ar_opt_hqc = ar_opt_hqc - 1;
ma_opt_hqc= ma_opt_hqc - 1;

% i.e. we need to choose from either ARMA(5,5) or ARMA(0,1)

%% Let's estimate our models
[EstMdl_aic,~,logL_aic] = estimate(arima(ar_opt_aic,0,ma_opt_aic),SP500.LogRet);
E_aic = infer(EstMdl_aic,SP500.LogRet); % estimated residuals from the estimated model
[EstMdl_bic,~,logL_bic] = estimate(arima(ar_opt_bic,0,ma_opt_bic),SP500.LogRet);
E_bic = infer(EstMdl_bic,SP500.LogRet); % estimated residuals from the estimated model
figure
subplot(1,2,1)
plot([SP500.LogRet(2:end), SP500.LogRet(2:end)-E_aic])
legend({'data','ARMA(5,5)'})
subplot(1,2,2)
plot([SP500.LogRet(2:end), SP500.LogRet(2:end)-E_bic])
legend({'data','ARMA(0,1)'})

%% Let's see what is going on with residuals

figure
subplot(2,3,1)
histfit(E_aic)
title("ARMA(5,5)")
subplot(2,3,2)
autocorr(E_aic,20)
subplot(2,3,3)
parcorr(E_aic,20)
subplot(2,3,4)
histfit(E_bic)
title("ARMA(0,1)")
subplot(2,3,5)
autocorr(E_bic,20)
subplot(2,3,6)
parcorr(E_bic,20)

%% ARCH-LM test

%%% step 2
[res_aic, con_var_aic] = infer(EstMdl_aic,SP500.LogRet); 
[res_bic, con_var_bic] = infer(EstMdl_bic,SP500.LogRet); 
%%% step 3
[h_arch_aic,pValue_arch_aic,stat_arch_aic,cValue_arch_aic] = archtest(res_aic, 'Lags',1:20);
[h_arch_bic,pValue_arch_bic,stat_arch_bic,cValue_arch_bic] = archtest(res_bic, 'Lags',1:20);
disp('ARCH test for residuals in ARMA(5,5)')
disp('Lags      Decision   LM-stat     p-val')
disp(string([(1:20)' h_arch_aic' stat_arch_aic' pValue_arch_aic']))
disp('ARCH test for residuals in ARMA(0,1)')
disp('Lags      Decision   LM-stat     p-val')
disp(string([(1:20)' h_arch_bic' stat_arch_bic' pValue_arch_bic']))

% The null hypothesis is rejected, i.e. we observe conditional heteroskedasticity

%% Ljung-Box test

%%% standardised residuals
z_aic = res_aic./sqrt(con_var_aic);
z_bic = res_bic./sqrt(con_var_bic);
%%% step 5
[h_lbq_aic,pValue_lbq_aic,stat_lbq_aic,cValue_lbq_aic] = lbqtest(z_aic,Lags=1:20);
[h_lbq_bic,pValue_lbq_bic,stat_lbq_bic,cValue_lbq_bic] = lbqtest(z_bic,Lags=1:20);
disp('Ljung-Box test for residuals in ARMA(5,5)')
disp('Decision   Lbq-stat     p-val')
disp(string([h_lbq_aic' stat_lbq_aic' pValue_lbq_aic']))
disp('Ljung-Box test for residuals in ARMA(0,1)')
disp('Decision   Lbq-stat     p-val')
disp(string([h_lbq_bic' stat_lbq_bic' pValue_lbq_bic']))

% Resiaduals of ARMA(5,5) follow the white noise process
% In other words, residuals of ARMA(0,1) exhibit autocrrelation
% We should stick to ARMA(5,5) as no autocorrelaion in residuals is an important assumption for GARCH-type models

%% Fit GARCH(1,1), EGARCH(1,1) sequentially (120, 252 rolling window), get the data on coefficients and fitted values

f=@(x) -GARCH11_LL(x,res_aic); %Anonymous function for fmincon (negative likelihood)
%set parameter constraints
A=[0 1 1];
B=1;
LB=[eps 0 0]; % eps is the smallest positive number different from zero
UB=Inf(1,3);
parm0 = [var(res_aic) 0.05  0.8]; %set initial value
thetahat=fmincon(f,parm0,A,B,[],[],LB,UB); %Estimating GARCH
disp('        omega     alpha      beta')
disp(thetahat) %display the GARCH estimated parameters
% alpha+beta is close to 1, highly persistent as expected
%%
[LL_hat,ht_hat,zt_hat]=GARCH11_LL(thetahat,res_aic); %estimating GARCH volatility
figure
subplot(1,3,1)
plot(SP500.Date(2:end),SP500.LogRet(2:end))
title("S&P 500 Log Return")
subplot(1,3,2)
plot(SP500.Date(2:end),SP500.LogRet(2:end).^2)
title("S&P 500 Log Return Squared")
subplot(1,3,3)
plot(SP500.Date(2:end),ht_hat)
title("GARCH(1,1) Fitted Conditional Variance for S&P 500")
% autocorr(ht_hat, 100) highly persistent

%% 

figure
subplot(1,2,1)
plot(SP500.Date(2:end),ht_hat)
title("GARCH(1,1) Fitted Conditional Variance for S&P 500")
subplot(1,2,2)
plot(SP500.Date(2:end),SP500.RV(2:end))
title("Ralised Variance for S&P 500")
ylim([0,0.004])

%%

figure
subplot(1,2,1)
plot(SP500.Date(2:end),SP500.LogRet(2:end))
title("S&P 500 Log return")
grid on
subplot(1,2,2)
plot(SP500.Date(2:end),ht_hat)
grid on
hold on
plot(SP500.Date(2:end),SP500.RV(2:end))
title("Variance")
legend({'GARCH(1,1) Conditional Variance','Realised Variance'})

%% GARCH sequentially with 2520 days estimation window

ew = 2500; % estimation window
gcoefs = zeros(length(SP500.RV)-ew-1,3);
frcst = zeros(length(SP500.RV)-ew-1,1);
for i = 1:length(SP500.RV)-ew-1
    % ARMA(5,5) first
    ind = i;
    [EstMdl,~,logL] = estimate(arima(ar_opt_aic,0,ma_opt_aic),SP500.LogRet(i+1:i+ew+1),'Display','off');
    [res, con_var] = infer(EstMdl,SP500.LogRet); 
    %set parameter constraints for GARCH(1,1)
    f=@(x) -GARCH11_LL(x,res); %Anonymous function for fmincon (negative likelihood)
    A=[0 1 1];
    B=1;
    LB=[eps 0 0]; % eps is the smallest positive number different from zero
    UB=Inf(1,3);
    parm0 = [var(res) 0.1  0.85]; %set initial value
    thetahat=fmincon(f,parm0,A,B,[],[],LB,UB); %Estimating GARCH
    gcoefs(i,1) = thetahat(1); % omega
    gcoefs(i,2) = thetahat(2); % alpha
    gcoefs(i,3) = thetahat(3); % beta
    [LL_hat,ht_hat,zt_hat]=GARCH11_LL(thetahat,res);
    frcst(i) = thetahat(1) + thetahat(2) * res(end).^2 + thetahat(3) * ht_hat(end);
end
%% 

figure
plot(frcst)

%%

figure
plot(ht_hat)
hold on
plot(SP500.RV(2:end))


%%
omega = [nan(ew+1,1); gcoefs(:,1)];
alpha = [nan(ew+1,1); gcoefs(:,2)];
beta = [nan(ew+1,1); gcoefs(:,3)];
gfrcst = [nan(ew+1,1); frcst];
ht_hat = [nan(1,1); ht_hat];
%%
SP500.GARCH_omega = omega;
SP500.GARCH_alpha = alpha;
SP500.GARCH_beta = beta;
SP500.GARCH_frcst = gfrcst;
SP500.ht_hat = ht_hat;

%%
figure
subplot(1,3,1)
plot(SP500.DY)
title("S&P 500 Dividend Yield")
subplot(1,3,2)
autocorr(SP500.DY, 100)
title("S&P 500 Dividend Yield Autocorrelattion")
subplot(1,3,3)
parcorr(SP500.DY, 100)
title("S&P 500 Dividend Yield Partial Autocorrelattion")

%%
figure
subplot(1,3,1)
plot(SP500.RF)
title("Risk-Free Rate")
subplot(1,3,2)
autocorr(SP500.RF, 100)
title("Risk-Free Rate Autocorrelattion")
subplot(1,3,3)
parcorr(SP500.RF, 100)
title("Risk-Free Rate Partial Autocorrelattion")
%% Daily, weekly, monthly realised volatilites

SP500 = SP500(~isnan(SP500.RV),:);
RV1 = zeros(length(SP500.RV)-22,1); % daily
RV5 = zeros(length(SP500.RV)-22,1); % weekly
RV22 = zeros(length(SP500.RV)-22,1); % monthly
for i = 23:length(SP500.RV)
    RV1(i-22) = sqrt(SP500.RV(i-1));
    RV5(i-22) = sqrt(sum(SP500.RV(i-5:i-1),"omitnan"));
    RV22(i-22) = sqrt(sum(SP500.RV(i-22:i-1),"omitnan"));
end

SP500.RV1 = [nan(22,1); RV1];
SP500.RV5 = [nan(22,1); RV5];
SP500.RV22 = [nan(22,1); RV22];

%% Finally regression coefficients

ew = 250;
HARcoefs = zeros(length(SP500.RV)-22-ew,5);
HARfrcts = zeros(length(SP500.RV)-22-ew,1);
for i = 23:length(SP500.RV)-ew
    X = [ones(ew,1) SP500.RV1(i:i+ew-1) SP500.RV5(i:i+ew-1) SP500.RV22(i:i+ew-1)];
    Y = sqrt(SP500.RV(i:i+ew-1));
    betas = X\Y;
    HARcoefs(i-22,1) = betas(1); % intercept
    HARcoefs(i-22,2) = betas(2); % daily beta
    HARcoefs(i-22,3) = betas(3); % weekly beta
    HARcoefs(i-22,4) = betas(4); % monthly beta
    HARcoefs(i-22,5) = var(Y - betas(1) - betas(2) * X(:,2) - betas(3) * X(:,3) - betas(4) * X(:,4)); % variance of the error term
    HARfrcts(i-22) = betas(1) + betas(2) * X(end,2) + betas(3) * X(end,3) + betas(4) * X(end,4);
end

%%

SP500.HAR_c = [nan(ew+22,1);HARcoefs(:,1)];
SP500.HAR_d = [nan(ew+22,1);HARcoefs(:,2)];
SP500.HAR_w = [nan(ew+22,1);HARcoefs(:,3)];
SP500.HAR_m = [nan(ew+22,1);HARcoefs(:,5)];
SP500.HAR_f = [nan(ew+22,1);HARfrcts];

%% And VIX

VIX_t = readtable("VIX.xlsx");
figure
plot(VIX_t.VIX)
SP500 = join(SP500,VIX_t,'Keys','Date');

%%

writetable(SP500,"Data_Prepared.xlsx")