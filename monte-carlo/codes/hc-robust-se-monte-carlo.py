import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t

# --------------------------------------------------
# Parameters
# --------------------------------------------------
R = 1000          # Monte Carlo iterations
n = 20            # sample size: 10, 20, 50
alpha = 0.05     # significance level

# --------------------------------------------------
# Storage
# --------------------------------------------------
se_hc0, se_hc1, se_hc2, se_hc3 = [], [], [], []
rej_hc0, rej_hc1, rej_hc2, rej_hc3 = [], [], [], []

# --------------------------------------------------
# Monte Carlo simulation
# --------------------------------------------------
for r in range(R):

    # 1. Data generating process
    z = np.random.normal(size=n)
    x = np.exp(z)
    eps = np.random.normal(size=n)
    y = x + np.sqrt(x) * eps

    df = pd.DataFrame({'y': y, 'x': x})

    # 2. OLS regression
    model = smf.ols('y ~ x', data=df).fit()
    beta1_hat = model.params['x']

    hc0 = model.HC0_se['x']
    hc1 = model.HC1_se['x']
    hc2 = model.HC2_se['x']
    hc3 = model.HC3_se['x']

    # 3. Store standard errors
    se_hc0.append(hc0)
    se_hc1.append(hc1)
    se_hc2.append(hc2)
    se_hc3.append(hc3)

    # 4. Hypothesis test: H0 = beta1 = 1
    crit = t.ppf(1 - alpha / 2, df=n - 2)

    rej_hc0.append(abs((beta1_hat - 1) / hc0) > crit)
    rej_hc1.append(abs((beta1_hat - 1) / hc1) > crit)
    rej_hc2.append(abs((beta1_hat - 1) / hc2) > crit)
    rej_hc3.append(abs((beta1_hat - 1) / hc3) > crit)

# --------------------------------------------------
# Results
# --------------------------------------------------
print('Mean standard errors (HC0–HC3):')
print(np.mean(se_hc0), np.mean(se_hc1),
      np.mean(se_hc2), np.mean(se_hc3))

print('\nRejection rates (HC0–HC3):')
print(np.mean(rej_hc0), np.mean(rej_hc1),
      np.mean(rej_hc2), np.mean(rej_hc3))
