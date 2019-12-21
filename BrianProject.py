#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arch
from statsmodels.tsa import arima_model
from scipy.optimize import minimize
# %%
f = '/mnt/d/study/2019/2707_Time_Series_Analysis_Statistical_Arbitrage/project/VIX.csv'
data = pd.read_csv(f, index_col=0)
data.head(5)
# %%
plt.plot(data['Close'])
plt.show()

# %%
close = data[['Close']]
train_idx = data.index.get_loc('2013-12-31')+1
val_idx = data.index.get_loc('2017-12-29')+1
close_train = close[:train_idx]
print(close_train.tail(2))
close_val = close[train_idx:val_idx]
print(close_val.tail(2))
close_test = close[val_idx:]
print(close_test.tail(2))
# %%
# ARIMA(p,d,q)
best_aic = np.inf
best_order = None
best_mdl = None
pq_rng = range(5) # [0,1,2,3,4]
d_rng = range(2) # [0,1]
for i in pq_rng:
    for d in d_rng:
        for j in pq_rng:
            try:
                tmp_mdl = arima_model.ARIMA(close_train, order=(i,d,j)).fit(method='mle', trend='nc')
                tmp_aic = tmp_mdl.aic
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (i, d, j)
                    best_mdl = tmp_mdl
            except: continue
print(f"best model (p,d,q): {best_order} with aic: {best_aic}" )
# %%
# ARIMA(1,1,1)-GARCH(1,1)
close_train_1Diff = close_train.diff()[1:]

# %%
# nabla Pt = kappa + beta nabla P_{t-1} + a_t + theta a_{t-1}
# a_t = sigma_t epsilon_t 
# sigma_t^2 = alpha_0 + alpha1 a_{t-1}^2 + alpha2 sigma_{t-1}^2
Xt = np.array(close_train_1Diff['Close'])
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
    return 0.5 * np.sum(np.log(vart)) + 0.5*np.sum(errorSum * errorSum/vart)

# %%
params = [np.mean(Xt),0.3, 0.3, np.std(Xt), 0.2,0.2,0.2]
res = minimize(ARIMA111_GARCH11_lhdfn, params, method = 'nelder-mead', 
        tol = 1e-5, options = {'disp': True, 'maxiter': 10000})
print(res.x)
# %%



# %%
