Monte Carlo Simulation: Robust Standard Errors (HC0–HC3)

This repository implements a Monte Carlo study of heteroskedasticity-robust
standard errors (HC0–HC3) in small samples.

The data-generating process features strong heteroskedasticity and high-leverage
observations. For different sample sizes, the code evaluates:

- Mean estimated standard errors
- Rejection rates of t-tests for H0: beta = 1

The results illustrate severe over-rejection of HC0–HC2 in small samples and
improved finite-sample performance of HC3.

