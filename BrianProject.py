#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arch
from statsmodels.tsa import arima_model
from scipy.optimize import minimize
# %%
f = '/mnt/d/study/2019/2707_Time_Series_Analysis_Statistical_Arbitrage/project/mine/VIX.csv'
data = pd.read_csv(f, index_col=0)
data.head(5)
# %%
plt.plot(np.array(data['Close']))
plt.show()
# %%
close = data[['Close']]
# train_idx = data.index.get_loc('2013-12-31')+1
# val_idx = data.index.get_loc('2017-12-29')+1
# close_train = close[:train_idx]
# print(close_train.tail(2))
# close_val = close[train_idx:val_idx]
# print(close_val.tail(2))
# close_test = close[val_idx:]
# print(close_test.tail(2))
#%%
def normal_lhdfn(mu, var):
    return 0.5 * np.sum(np.log(var)) + 0.5*np.sum(mu * mu/var)
# %%
# ARIMA(1,1,1)
def ARIMA111(Xt, days):
    Xt = Xt[1:] - Xt[:-1]
    def ARIMA111_lhdfn(params):
        alpha1 = params[0]
        beta1 = params[1]
        variable = Xt[1:] - alpha1 * Xt[:-1]
        return normal_lhdfn(variable, 1+beta1*beta1)
    params = [0.2,0.2]
    res = minimize(ARIMA111_lhdfn, params, method = 'nelder-mead', 
            tol = 1e-8, options = {'disp': True, 'maxiter': 10000})
    return res
# %%
# nabla Pt = kappa + beta nabla P_{t-1} + a_t + theta a_{t-1}
# a_t = sigma_t epsilon_t 
# sigma_t^2 = alpha_0 + alpha1 a_{t-1}^2 + alpha2 sigma_{t-1}^2
def ARIMA111_GARCH11(Xt):
    Xt = Xt[1:] - Xt[:-1]
    def ARIMA111_GARCH11_lhdfn(params):
        kappa = params[0] 
        beta = params[1]
        theta = params[2]
        sigma = params[3]
        alpha0 = params[4]
        alpha1 = params[5]
        alpha2 = params[6]
        errorSum = Xt[1:] - kappa  - beta * Xt[:-1]
        at = np.zeros(len(Xt))
        at[0] = errorSum[0]
        sigmat_squared = np.zeros(len(Xt))
        sigmat_squared[0] = sigma * sigma
        for i in range(1,len(Xt)):
            sigmat_squared[i] = alpha0 + alpha1 * at[i-1]*at[i-1] + alpha2 * sigmat_squared[i-1]
            at[i] = errorSum[i-1] - theta * at[i-1]
        vart = np.zeros(len(Xt) - 1)
        for i in range(0,len(vart)):
            vart[i] = sigmat_squared[i+1] + theta * theta * sigmat_squared[i]
        return normal_lhdfn(errorSum,vart)
    params = [np.mean(Xt),0.3, 0.3, np.std(Xt), 0.2,0.2,0.2]
    res = minimize(ARIMA111_GARCH11_lhdfn, params, method = 'nelder-mead', 
            tol = 1e-8, options = {'disp': True, 'maxiter': 10000})
    return res


# %%
def ARIMA111_GJRGARCH11(Xt):
    Xt = Xt[1:] - Xt[:-1]
    def ARIMA111_GJRGARCH11_lhdfn(params):
        kappa = params[0] 
        beta = params[1]
        theta = params[2]
        sigma = params[3]
        alpha0 = params[4]
        alpha1 = params[5]
        alpha2 = params[6]
        gamma1 = params[7]
        errorSum = Xt[1:] - kappa  - beta * Xt[:-1]
        at = np.zeros(len(Xt))
        at[0] = errorSum[0]
        sigmat_squared = np.zeros(len(Xt))
        sigmat_squared[0] = sigma * sigma
        for i in range(1,len(Xt)):
            sigmat_squared[i] = alpha0 + alpha1 * at[i-1]*at[i-1] + alpha2 * sigmat_squared[i-1] 
            if at[i-1] > 0:
                sigmat_squared[i] += gamma1 * at[i-1] * at[i-1]
            at[i] = errorSum[i-1] - theta * at[i-1]
        vart = np.zeros(len(Xt) - 1)
        for i in range(0,len(vart)):
            vart[i] = sigmat_squared[i+1] + theta * theta * sigmat_squared[i]
        return normal_lhdfn(errorSum,vart)
    params = [np.mean(Xt),0.3, 0.3, np.std(Xt), 0.2,0.2,0.2]
    res = minimize(ARIMA111_GJRGARCH11_lhdfn, params, method = 'nelder-mead', 
            tol = 1e-8, options = {'disp': True, 'maxiter': 10000})
    return res
# %%
def heston(Xt):
    def heston_lhdfn(params):
        lam = params[0]
        mu = params[1]
        sigma = params[2]
        dXt = Xt[1:] - Xt[:-1]
        variable = dXt + lam * (Xt[:-1] - mu) 
        return normal_lhdfn(variable, sigma**2)
    params = [0.2, np.mean(Xt), np.var(Xt)]
    res = minimize(heston_lhdfn, params, method = 'nelder-mead', 
            tol = 1e-8, options = {'disp': True, 'maxiter': 10000})
    return res
# %%
def OU(Xt):
    def OU_lhdfn(params):
        lam = params[0]
        mu = params[1]
        sigma = params[2]
        lnXt = np.log(Xt)
        lnXt_diff = lnXt[1:] - lnXt[:-1]
        variable = lnXt_diff + lam* (lnXt[:-1] - mu)
        return normal_lhdfn(variable, sigma**2)
    params = [0.2, np.mean(Xt), np.var(Xt)]
    res = minimize(OU_lhdfn, params, method = 'nelder-mead', 
            tol = 1e-8, options = {'disp': True, 'maxiter': 10000})
    return res
# %%
f_future = '/mnt/d/study/2019/2707_Time_Series_Analysis_Statistical_Arbitrage/project/mine/VIX_Future.csv'
data_future = pd.read_csv(f_future, index_col=0)
data_future.head(5)

# %%
date = data_future.index[0]
t_date = close.index.get_loc(date)
best_order, best_mdl, best_aic = ARIMA111(np.array(close[:t_date]))

# %%
print(close.iloc[t_date])
for i in range(5):
    print(best_mdl.forecast(steps = 5))

# %%
