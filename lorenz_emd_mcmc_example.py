# Teo Price-Broncucia 2020

from scipy.integrate import solve_ivp
import chaos_utils as cu
import numpy as np
import emd_mcmc as em
import matplotlib.pyplot as plt
from scipy.stats import uniform
import os

# Generate reference points
ic = [8, 13, 13]

truth_args = [10, 28, 8/3]

max_t = 50

t_span = (0, max_t)

eval_ts = np.linspace(0, max_t, num=1000)

truth = solve_ivp(cu.lorenz_deriv, t_span, ic, args=truth_args, rtol=1e-3, t_eval=eval_ts)

# ode_return.y output is (dimensions x points)
truth_points = truth.y.T

# Set random seed for repeatability
np.random.seed(1000)

ic_noise = np.random.normal(size = 3)

print(f"truth ic: {truth_points[0, :]}")
truth_points[0, :] = truth_points[0, :] + ic_noise
print(f"perturbed truth ic: {truth_points[0, :]}")

# define priors

# Gaussian Priors, must be passed as tuple of mu and covariance matrix
arg_mus = [9, 25, 5]
arg_cov = np.identity(3)*10

# Uniform Priors
uniform_priors = uniform(loc=[0, 0, 0], scale = 30)

transition_q_mus = [0, 0, 0]
transition_q_cov = np.identity(3)*8

chain_len = 500
burn_in = 400

# Case 1 Gaussian Priors, EMD MCMC
# chain, rejected = em.emd_mcmc(truth_points, cu.lorenz_deriv, t_span, eval_ts, (arg_mus, arg_cov), (transition_q_mus, transition_q_cov), chain_len, burn_in)

# Case 2 Uniform Priors, EMD MCMC
# chain, rejected = em.emd_mcmc(truth_points, cu.lorenz_deriv, t_span, eval_ts, uniform_priors, (transition_q_mus, transition_q_cov), chain_len, burn_in,uniform=True)

# Case 3 Gaussian Priors, Pointwise MCMC
chain, rejected = em.pointwise_mcmc(truth_points, cu.lorenz_deriv, t_span, eval_ts, (arg_mus, arg_cov), (transition_q_mus, transition_q_cov), chain_len, burn_in)

# Case 4 Unifrom Priors, Pointwise MCMC
# chain, rejected = em.pointwise_mcmc(truth_points, cu.lorenz_deriv, t_span, eval_ts, uniform_priors, (transition_q_mus, transition_q_cov), chain_len, burn_in,uniform=True)

chain = np.array(chain)
rejected = np.array(rejected)

fig, axs = plt.subplots(3, figsize=(8, 8))

axs[0].hist(chain[:, 0], bins=30)
axs[0].set_title(r"$\sigma$")
axs[1].hist(chain[:, 1], bins=30)
axs[1].set_title(r"$\rho$")
axs[2].hist(chain[:, 2], bins=30)
axs[2].set_title(r"$\beta$")

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Parameter Value", fontsize=18)
plt.ylabel("Markov Chain Points", fontsize=18)
plt.subplots_adjust(top = 0.90, bottom=0.1, hspace=.3, wspace=0.4)

if not os.path.isdir('figures'):
    print('not found')
    os.mkdir('figures')

# Case 1
# fig.suptitle("Gaussian Priors EMD Likelihood Parameter Distributions", fontsize=18)
# plt.savefig(f"figures/mcmc_emd_gaussian_{chain_len}_{burn_in}")

# Case 2
# fig.suptitle("Uniform Priors EMD Likelihood Parameter Distributions", fontsize=18)
# plt.savefig(f"figures/mcmc_emd_uniform_{chain_len}_{burn_in}")

# Case 3
fig.suptitle("Gaussian Priors Pointwise Likelihood Parameter Distributions", fontsize=18)
plt.savefig(f"figures/mcmc_pw_gaussian_{chain_len}_{burn_in}")

# Case 4
# fig.suptitle("Uniform Priors Pointwise Likelihood Parameter Distributions", fontsize=18)
# plt.savefig(f"figures/mcmc_pw_uniform_{chain_len}_{burn_in}")
