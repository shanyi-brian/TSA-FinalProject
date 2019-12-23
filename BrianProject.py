#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arch
from statsmodels.tsa import arima_model
from scipy.optimize import minimize
from scipy.stats import norm
# %%
f = '/mnt/d/study/2019/2707_Time_Series_Analysis_Statistical_Arbitrage/project/mine/VIX.csv'
data = pd.read_csv(f, index_col=0)
data.head(5)
# %%
plt.plot(np.array(data['Close']))
plt.show()
# %%
close = data[['Close']]
train_idx = data.index.get_loc('2013-12-31')+1
val_idx = data.index.get_loc('2017-12-29')+1
close_train = close[:train_idx]
print(close_train.tail(2))
close_val = close[(train_idx-1):val_idx]
print(close_val.tail(2))
#%%
def normal_lhdfn(mu, var):
    if len(var) == 1:
        return np.log(var) * len(mu) + np.sum(mu * mu/var)
    return np.sum(np.log(var)) + np.sum(mu * mu/var)
# ARIMA(1,1,1)
# def ARIMA111(Xt):
#     Xt = Xt[1:] - Xt[:-1]
#     def ARIMA111_lhdfn(params):
#         alpha1 = params[0]
#         beta1 = params[1]
#         variable = Xt[1:] - alpha1 * Xt[:-1]
#         return normal_lhdfn(variable, 1+beta1*beta1)
#     params = [0.2,0.2]
#     res = minimize(ARIMA111_lhdfn, params, method = 'nelder-mead', 
#             tol = 1e-8, options = {'disp': True, 'maxiter': 10000})
#     return res

# def ARIMA111_pred(Xt, params, days, sims_per_days = 10):
#     Xt_diff = Xt[1:] - Xt[:-1]
#     Z = Xt_diff[0]
#     alpha1 = params[0]
#     beta1 = params[1]
#     variable = Xt_diff[1:] - alpha1 * Xt_diff[:-1]
#     for i in range(variable):
#         Z = variable - beta1 * Z
#     Z = [Z]
#     X = [Xt_diff[-1]]
#     for d in range(days):
#         assert(len(X) == len(Z))
#         new_Z = []
#         new_X = []
#         for i in range(len(Z)):
#             Zt = np.random.normal(size = sims_per_days)
#             new_Z.append(Zt)
#             new_X.append(alpha1 * X)
# %%
def ARIMA(Xt):
    Xt = np.array(Xt)
    best_aic = np.inf
    best_order = None
    best_mdl = None
    pq_rng = range(5) # [0,1,2,3,4]
    d_rng = range(2) # [0,1]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = arima_model.ARIMA(Xt, order=(i,d,j)).fit(method='mle', trend='nc')
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue
    return best_mdl,best_order
# %%
# nabla Pt = kappa + beta nabla P_{t-1} + a_t + theta a_{t-1}
# a_t = sigma_t epsilon_t 
# sigma_t^2 = alpha_0 + alpha1 a_{t-1}^2 + alpha2 sigma_{t-1}^2
def ARIMA111_GARCH11_lhdfn(Xt, params, a0 = None):
        kappa = params[0] 
        beta = params[1]
        theta = params[2]
        sigma = params[3]
        alpha0 = params[4]
        alpha1 = params[5]
        alpha2 = params[6]
        errorSum = Xt[1:] - kappa  - beta * Xt[:-1]
        at = np.zeros(len(Xt))
        if a0 is None:
            at[0] = errorSum[0]
        else:
            at[0] = a0
        sigmat_squared = np.zeros(len(Xt))
        sigmat_squared[0] = sigma * sigma
        for i in range(1,len(Xt)):
            sigmat_squared[i] = alpha0 + alpha1 * at[i-1]*at[i-1] + alpha2 * sigmat_squared[i-1]
            at[i] = errorSum[i-1] - theta * at[i-1]
        vart = np.zeros(len(Xt) - 1)
        for i in range(0,len(vart)):
            vart[i] = sigmat_squared[i+1] + theta * theta * sigmat_squared[i]
        return normal_lhdfn(errorSum,vart)

def ARIMA111_GARCH11(Xt):
    Xt = Xt[1:] - Xt[:-1]
    minimize_fn = lambda x: ARIMA111_GARCH11_lhdfn(Xt,x)
    params = [np.mean(Xt),0.4, 0.4, np.std(Xt), 0.4,0.4,0.4]
    res = minimize(minimize_fn, params, method = 'nelder-mead', 
            tol = 1e-8, options = {'disp': True, 'maxiter': 10000})
    print("Model trained finished")
    # update sigmat
    params = res.x
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
    params[3] = np.sqrt(sigmat_squared[-1])
    return params, at[-1]

# %%
def ARIMA111_GJRGARCH11_lhdfn(Xt, params, a0 = None):
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
        if a0 is None:
            at[0] = errorSum[0]
        else:
            at[0] = a0
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

def ARIMA111_GJRGARCH11(Xt):
    Xt = Xt[1:] - Xt[:-1]
    params = [np.mean(Xt),0.4, 0.4, np.std(Xt), 0.4,0.4,0.4, 0.4]
    minimize_fn = lambda x: ARIMA111_GJRGARCH11_lhdfn(Xt,x)
    res = minimize(minimize_fn, params, method = 'nelder-mead', 
            tol = 1e-8, options = {'disp': True, 'maxiter': 10000})
    params = res.x
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
    params[3] = np.sqrt(sigmat_squared[-1])
    return params, at[-1]
# %%
def heston_lhdfn(Xt, params):
        lam = params[0]
        mu = params[1]
        sigma = params[2]
        dXt = Xt[1:] - Xt[:-1]
        variable = dXt + lam * (Xt[:-1] - mu) 
        return normal_lhdfn(variable, [sigma**2])

def heston(Xt):
    minimize_fn = lambda x: heston_lhdfn(Xt,x)
    params = [0.2, np.mean(Xt), np.var(Xt)]
    res = minimize(minimize_fn, params, method = 'nelder-mead', 
            tol = 1e-8, options = {'disp': True, 'maxiter': 10000})
    return res.x
# %%
def OU_lhdfn(Xt, params):
        lam = params[0]
        mu = params[1]
        sigma = params[2]
        lnXt = np.log(Xt)
        lnXt_diff = lnXt[1:] - lnXt[:-1]
        variable = lnXt_diff + lam* (lnXt[:-1] - mu)
        return normal_lhdfn(variable, [sigma**2])

def OU(Xt):
    minmize_fn = lambda x: OU_lhdfn(Xt,x)
    params = [0.2, np.mean(Xt), np.var(Xt)]
    res = minimize(minmize_fn, params, method = 'nelder-mead', 
            tol = 1e-8, options = {'disp': True, 'maxiter': 10000})
    return res.x

# %%
def ARIMA_AIC(Xt, params, order):
    p = order[0]
    d = order[1]
    q = order[2]
    for i in range(d):
        Xt = Xt[1:] - Xt[:-1]
    variable = Xt[p:]
    for i in range(p):
        variable -= params[i] * Xt[(p-1-i):(-i-1)]
    variance = 1
    for i in range(q):
        variance += params[-(i+1)] ** 2
    return normal_lhdfn(variable, [variance]) + 2 * (p+q)

def ARIMA111_GARCH11_AIC(Xt, params, at):
    return ARIMA111_GARCH11_lhdfn(Xt, params,at) + 2 * len(params)

def ARIMA111_GJRGARCH11_AIC(Xt, params, at):
    return ARIMA111_GJRGARCH11_lhdfn(Xt, params,at) + 2*(len(params))

def heston_AIC(Xt, params):
    return heston_lhdfn(Xt, params) + 2 * (len(params))

def OU_AIC(Xt, params):
    return OU_lhdfn(Xt, params) + 2 * (len(params))

#%%
md_ARIMA,order_ARIMA = ARIMA(np.array(close_train))
print(md_ARIMA.summary())
ARIMA_train_AIC= ARIMA_AIC(np.array(close_train), md_ARIMA.params, order_ARIMA)
ARIMA_val_AIC = ARIMA_AIC(np.array(close_val), md_ARIMA.params, order_ARIMA)
print(ARIMA_train_AIC)
print(ARIMA_val_AIC)
#%%
params, at = ARIMA111_GARCH11(np.array(close_train))
ARIMAGARCH_train_AIC= ARIMA111_GARCH11_AIC(np.array(close_train), md_ARIMA.params, at)
ARIMAGARCH_val_AIC = ARIMA111_GARCH11_AIC(np.array(close_val), params,at)
print(ARIMAGARCH_train_AIC)
print(ARIMAGARCH_val_AIC)
# %%
params, at = ARIMA111_GJRGARCH11(np.array(close_train))
ARIMAGJRGARCH_train_AIC = ARIMA111_GJRGARCH11_AIC(np.array(close_train), params,at)
ARIMAGJRGARCH_error_AIC = ARIMA111_GJRGARCH11_AIC(np.array(close_val), params,at)
print(ARIMAGJRGARCH_train_AIC)
print(ARIMAGJRGARCH_error_AIC)

# %%
params = heston(np.array(close_train))
heston_train_AIC = heston_AIC(np.array(close_train),params)
heston_error_AIC = heston_AIC(np.array(close_val),params)
print(heston_train_AIC)
print(heston_error_AIC)

#%%
params = OU(np.array(close_train))
OU_train_AIC =OU_AIC(np.array(close_train),params)
OU_error_AIC = OU_AIC(np.array(close_val),params)
print(OU_train_AIC)
print(OU_error_AIC)
# %%
f_future = '/mnt/d/study/2019/2707_Time_Series_Analysis_Statistical_Arbitrage/project/mine/VIX_Future.csv'
data_future = pd.read_csv(f_future, index_col=0)
data_future.head(5)

# def ARIMA_profit(p = 0.2):
#     ret = []
#     for date, row in data_future.iterrows(): 
#         t_date = close.index.get_loc(date)
#         best_mdl, conf = ARIMA(np.array(close[:t_date]), p)
#         settle_price = row['Settle']
#         maturity_settle_price = row['Maturity_Settle']
#         if settle_price < conf[0]:
#             profit = (maturity_settle_price - settle_price)/settle_price
#         elif settle_price > conf[1]:
#             profit = (settle_price - maturity_settle_price)/settle_price
#         else:
#             profit = 0
#         ret.append(profit)
#     data_future['ARIMA_return'] = ret 
# ARIMA_profit()
# %%

