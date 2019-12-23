1. ARIMA(p,d,q)
$$
	\nabla^d X_t = \alpha_1 \nabla^d X_{t-1} + \cdots + \alpha_p \nabla^d X_{t-p} + Z_t + \beta_1 Z_{t-1}+ \cdots + \beta_q Z_{t-q}
$$
Hence, $\nabla^d X_t - \alpha_1 \nabla^d X_{t-1} + \cdots + \alpha_p \nabla^d X_{t-p} \sim N(0,1+\beta_1^2+ \cdots + \beta_q^2)$

2. ARIMA(1,1,1)-GARCH(1,1)

	Let $P_t = \nabla X_t, P_0 = \kappa + a_0$ where $a_0 = \sigma_0 \varepsilon_0$
$$
P_t = \kappa + \beta_1 P_{t-1} + a_t + \theta a_{t-1}
$$
$$
a_t = \sigma_t \varepsilon_t \qquad \qquad \sigma_t^2 = \alpha_0 + \alpha_1 a_{t-1}^2 + \alpha_2 \sigma_{t-1}^2
$$
Hence, $P_t - \kappa - \beta_1P_{t-1} -  \theta a_{t-1} \sim N(0, \sigma_t^2)$.

1. Heston model
$$
dVIX_t  = \lambda (\mu  - VIX_t) * dt + \sigma \times \sqrt{VIX_t} * dW_t
$$
Hence, $\Delta VIX_t - \lambda (\mu - VIX_t) \sim N(0, \sigma^2\sqrt{VIX_t})$


4. OU process
$$
d \ln VIX_t = \lambda * (\mu - \ln VIX_t) * dt + \sigma * dW_t
$$

5. GJR-GARCH model
$$
\sigma_t^2 = \alpha_0 + \alpha_1 a_{t-1}^2 + \gamma_1 I_{a_{t-1}}a_{t-1}^2 + \alpha_2 \sigma_{t-1}^2
$$

Note that $I_{a_{t-1}} = \begin{cases} 1 & \text{if } a_{t-1} > 0 \\ 0 &\text{otherwise} \end{cases}$ which is the inverse of regular GJR-GARCH