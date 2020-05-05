# Teo Price-Broncucia 2020

import numpy as np
import ot
from scipy.stats import multivariate_normal as mv_norm
from scipy.integrate import solve_ivp
from scipy.stats import uniform
import chaos_utils as cu

# Earthmovers distance setup
# Provide needed calculated values from true distribution that will be resued
# a_points : true sampled points (points * dimensions) by default should be generated data
# bins = 11 : number of bins
# Returns:
# a_hist, bin_edges, distance_M
def emd_setup(a_points, bins=10):
    _, dim = a_points.shape
    
    a_hist, a_edges = np.histogramdd(a_points, bins=bins, density=True)

    midpoints = []
    for edge in a_edges:
        midpoints.append((edge[:-1] + edge[1:])/2)

    gridded = np.meshgrid(*midpoints)

    stacked = np.stack(gridded, axis=dim).reshape(-1, dim)

    M = ot.dist(stacked, metric="euclidean")

    return a_hist, a_edges, M

# Earthmovers distance likelihood
# a_hist : histogram calculated from a
# b_points : array like distribution (points * dimensions) by default should be generated data
# bins_edges : dimensions * arrays : bin_edges in each dimension for calucation of EMD, should be generated and supplied from true data
# distance_M : distance matrix precalculated from bins 
# t : likelihood tuneable
# Returns:
# likelihood
def emd_likelihood(a_hist, b_points, bin_edges, distance_M, t=7):
    # calculate histogram density of b using supplied bin edges
    b_hist, _ = np.histogramdd(b_points, bins=bin_edges, density=True)

    # maybe move reshape of a out since it will be getting done a lot
    emd = ot.emd2(a_hist.reshape(-1), b_hist.reshape(-1), distance_M)
    likelihood = 1 / (t * emd + 1)

    return likelihood


# Metropolis Hastings acceptance w/ symmetric proposal
def mcmc_accept(theta_old, theta_new, old_likliehood, new_likelihood, prior_dist, uniform=False):
    if uniform:
        prior_new = prior_dist.pdf(theta_new).prod()
        if prior_new == 0:
            P_new = np.NINF
            return False
        else:
            P_new = np.log(new_likelihood) + np.log(prior_new)
        
        prior_old = prior_dist.pdf(theta_old).prod()
        if prior_old == 0:
            P_old = np.NINF
        else:
            P_old = np.log(old_likliehood) + np.log(prior_old)

    else:
        P_new = np.log(new_likelihood) + np.log(prior_dist.pdf(theta_new))
        
        P_old = np.log(old_likliehood) + np.log(prior_dist.pdf(theta_old))

    if P_new > P_old:
        return True

    else:
        u = np.random.uniform(0, 1)
        if (u < np.exp(P_new - P_old)):
            return True
        else:
            return False

# emd_mcmc wrapper function
def emd_mcmc(truth_points, func, t_span, eval_ts, priors, transitions, iterations, burn_in, uniform=False):
    # # Define distribution for argument priors

    # Try with uniform priors
    if uniform:
        prior_dist = priors

    else:
        # # Define distribution for argument priors
        arg_mus, arg_cov = priors
        prior_dist = mv_norm(mean=arg_mus, cov=arg_cov)

    #define distribution for transitions
    transition_q_mus, transition_q_cov = transitions
    transition_dist = mv_norm(mean=transition_q_mus, cov=transition_q_cov)

    # generate initial values for args
    negative_check = True
    while negative_check:
        ic_args = prior_dist.rvs()
        if ic_args.min() > 0:
            negative_check = False
    print(f"Starting Arguments: {ic_args}")

    ic_point = truth_points[0, :]
    print(f"Starting point: {ic_point}")

    truth_hist, truth_bins, M = emd_setup(truth_points)

    theta_old = ic_args
    old_output = solve_ivp(func, t_span, ic_point, args=theta_old, rtol=1e-3, t_eval=eval_ts)
    old_output_points = old_output.y.T

    old_likelihood = emd_likelihood(truth_hist, old_output_points, truth_bins, M)
    print(f"Initial Likelihood: {old_likelihood}")

    chain = []
    rejected = []
    chain.append(theta_old)

    i = 0
    while i < iterations:
        # propose new point
        negative_check = True
        while negative_check:
            theta_new = theta_old + transition_dist.rvs()
            if theta_new.min() > 0:
                negative_check = False
        
        new_output = solve_ivp(func, t_span, ic_point, args=theta_new, rtol=1e-3, t_eval=eval_ts, max_step=10)
        new_output_points = new_output.y.T
        new_likelihood = emd_likelihood(truth_hist, new_output_points, truth_bins, M)

        if uniform:
            accepted = mcmc_accept(theta_old, theta_new, old_likelihood, new_likelihood, prior_dist, uniform=True)
        else:
            accepted = mcmc_accept(theta_old, theta_new, old_likelihood, new_likelihood, prior_dist)

        if accepted:
            chain.append(theta_new)
            theta_old = theta_new
            old_likelihood = new_likelihood
            i += 1

        else:
            rejected.append(theta_new)

    print(f"acceptance ratio: {len(chain)/(len(rejected) + len(chain))}")
    return chain[burn_in:], rejected[burn_in:]

# pointwise likelihood
def pw_likelihood(a_points, b_points, t=1):
    # calculate histogram density of b using supplied bin edges
    pw_dist = cu.pointwise_distance_sum(a_points, b_points)/len(a_points)

    likelihood = 1 / (t * pw_dist + 1)

    return likelihood

# pointwise wrapper function
def pointwise_mcmc(truth_points, func, t_span, eval_ts, priors, transitions, iterations, burn_in, uniform=False):
    # # Define distribution for argument priors

    # Uniform priors
    if uniform:
        prior_dist = priors

    else:
        # Define distribution for argument priors
        arg_mus, arg_cov = priors
        prior_dist = mv_norm(mean=arg_mus, cov=arg_cov)

    #define distribution for transitions
    transition_q_mus, transition_q_cov = transitions
    transition_dist = mv_norm(mean=transition_q_mus, cov=transition_q_cov)

    # generate initial values for args
    negative_check = True
    while negative_check:
        ic_args = prior_dist.rvs()
        if ic_args.min() > 0:
            negative_check = False
    print(f"Starting Arguments: {ic_args}")

    ic_point = truth_points[0, :]
    print(f"Starting point: {ic_point}")

    theta_old = ic_args
    old_output = solve_ivp(func, t_span, ic_point, args=theta_old, rtol=1e-3, t_eval=eval_ts)
    old_output_points = old_output.y.T

    old_likelihood = pw_likelihood(truth_points, old_output_points)
    print(f"Initial Likelihood: {old_likelihood}")

    chain = []
    rejected = []
    chain.append(theta_old)

    i = 0
    while i < iterations:
        # propose new point
        negative_check = True
        while negative_check:
            theta_new = theta_old + transition_dist.rvs()
            if theta_new.min() > 0:
                negative_check = False
        
        new_output = solve_ivp(func, t_span, ic_point, args=theta_new, rtol=1e-3, t_eval=eval_ts, max_step=10)
        new_output_points = new_output.y.T
        new_likelihood = pw_likelihood(truth_points, new_output_points)

        if uniform:
            accepted = mcmc_accept(theta_old, theta_new, old_likelihood, new_likelihood, prior_dist, uniform=True)
        else:
            accepted = mcmc_accept(theta_old, theta_new, old_likelihood, new_likelihood, prior_dist)

        if accepted:
            chain.append(theta_new)
            theta_old = theta_new
            old_likelihood = new_likelihood
            i += 1

        else:
            rejected.append(theta_new)

    print(f"acceptance ratio: {len(chain)/(len(rejected) + len(chain))}")
    return chain[burn_in:], rejected[burn_in:]