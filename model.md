1. ARIMA(p,d,q)
$$
	\nabla^d X_t = \alpha_1 \nabla^d X_{t-1} + \cdots + \alpha_p \nabla^d X_{t-p} + Z_t + \beta_1 Z_{t-1}+ \cdots + \beta_q Z_{t-q}
$$

2. ARIMA(1,1,1)-GARCH(1,1)

Let $P_t = \nabla^1 X_t$
$$
P_t = \kappa + \beta_1 P_{t-1} + a_t + \theta a_{t-1}
$$
$$
a_t = \sigma_t \varepsilon_t \qquad \qquad \sigma_t^2 = \alpha_0 + \alpha_1 a_{t-1}^2 + \alpha_2 \sigma_{t-1}^2
$$
Hence, $P_t - \kappa - \beta_1P_{t-1} \sim N(0, \sigma_t^2 + \theta^2 \sigma_{t-1}^2)$.