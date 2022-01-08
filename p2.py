import numpy as np
import matplotlib.pyplot as plt

mu_c = 0.0189
sig_c = 0.015

#mu_d = 0.01287
mu_d = mu_c
sig_d = 0.112

rho = 0.2
gamma = 2
beta = 0.98

A = beta*np.exp(mu_d - gamma*mu_c + (gamma*sig_c - rho*sig_d)**2/2 + sig_d**2*(1 - rho**2)/2)

f = A/(1-A)


Rf = np.exp(gamma*mu_c - (gamma**2*sig_c**2))/beta - 1
R = (1+f)/f - 1

gamma_seq = np.arange(0, 10, 0.01)


A_seq = beta*np.exp(mu_d - gamma_seq*mu_c + (gamma_seq*sig_c - rho*sig_d)**2/2 + sig_d**2*(1 - rho**2))

f_seq = A_seq/(1-A_seq)


Rf_seq = np.exp(gamma_seq*mu_c - (gamma_seq**2*sig_c**2))/beta - 1
R_seq = (1+f_seq)/f_seq - 1

plt.plot(gamma_seq, R_seq); plt.ylabel('Gross Returns'); plt.xlabel('Gamma'); plt.show()

plt.plot(gamma_seq, R_seq - Rf_seq); plt.ylabel('Gross Excess Returns'); plt.xlabel('Gamma'); plt.show()
