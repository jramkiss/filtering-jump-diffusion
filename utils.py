"""
This file contains utility functions for our experiments such as: 
     - Functions to help with plots
     - The resampling function described in the paper
     - RMSE calculation
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
from jax import lax
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pfjax as pf


def plot_particles (x_state, y_meas, vol_particles, price_particles, n_res, n_obs, point_plot, title="", plot_res=True):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex = True)
    fig.suptitle(title)

    sns.lineplot(data= x_state[..., 1].reshape(n_res*n_obs, 1)[(n_res-1):].squeeze(),
                 ax = ax[0], linewidth = 0.7,
                 label = "Latent")
    sns.scatterplot(x = "Time", y = "Log Asset Price", 
                    data = point_plot,
                    color = "firebrick",
                    ax = ax[0],
                    label="Observed").set(xlabel="Time",title ="$X_t$");

    sns.lineplot(data= x_state[..., 0].reshape(n_res*n_obs,1)[(n_res-1):].squeeze(),
                 ax = ax[1], linewidth = 0.7,
                 label = "Latent Volatility").set(xlabel="Time",title = "$Z_t$", ylabel="Volatility")
    if plot_res:
        for t in range(n_obs-1):
            for s in range(n_res):
                my_x = (t*n_res) + s
                sns.scatterplot(x = my_x, 
                                y=price_particles[t, :, s], 
                                s = 2, alpha = 0.5,
                                color = "green", ax = ax[0]);
                sns.scatterplot(x = my_x, 
                                y=vol_particles[t, :, s], 
                                s = 2, alpha = 0.5,
                                color = "green", ax = ax[1]);
    else:
        for t in range(n_obs-1):
            sns.scatterplot(x = t*n_res, 
                            y=price_particles[t, :, n_res-2], 
                            s = 2, alpha = 0.5,
                            color = "green", ax = ax[0]);
            sns.scatterplot(x = t*n_res, 
                            y=vol_particles[t, :, n_res-2], 
                            s = 2, alpha = 0.5,
                            color = "green", ax = ax[1]);
            
            
def jittered_multinomial(key, x_particles_prev, logw, h):
    r"""
    Variant of the resampling function presented in: 
    https://www.tandfonline.com/doi/abs/10.1198/jcgs.2009.07137
    
    They jitter each particle with N(0, h^2 B). However it is not clear how B is selected.
    I have selected the kernel bandwidth here, h, as 1/num_particles
    """
    prob = pf.utils.logw_to_prob(logw)
    n_particles = logw.size
    ancestors = random.choice(key,
                              a=jnp.arange(n_particles),
                              shape=(n_particles,), p=prob)
    jitter = random.normal(key, shape = x_particles_prev.shape) * h
    return {
        "x_particles": x_particles_prev[ancestors, ...] + jitter,
        "ancestors": ancestors
    }


def rmse (x, y):
    """ Find RMSE between x and y """
    return jnp.sqrt(jnp.mean((x-y)**2))


def quantile_index (logw, q):
    """
    Returns the index of the q-th quantile of logw
    """
    w = pf.utils.logw_to_prob(logw)
    val = jnp.quantile(w, q=q)
    nearest_ind = jnp.argmin(jnp.abs(val - w)) # find index of closest point to val
    return nearest_ind
    

def pf_timer (pf_method, n_particles_list, n_sim=15):
    r"""
    Function for recording the runtime of different particle filters for a given number of particles. Each
    setting of n_particles_list is evaluated n_sim times
    
    Args: 
        - pf_method: Partial function accepting n_particles and running the particle filter one time
        - n_particles_list: list of number of particles to evaluate the particle filter with
    
    Returns: 
        - adlfkj
    """
    all_times = jnp.zeros((len(n_particles_list), n_sim))
    all_loglik = jnp.zeros((len(n_particles_list), n_sim))
    for i, n_particles in tqdm(enumerate(n_particles_list)):
        for sim_number in range(n_sim):
            start = time.perf_counter()
            loglik = pf_method(n_particles=n_particles)
            end = time.perf_counter()
            all_times = all_times.at[i, sim_number].set(end - start) # record runtime 
            all_loglik = all_loglik.at[i, sim_number].set(loglik["loglik"]) # record loglik for my sanity
    return {
        "all_times": all_times,
        "all_loglik": all_loglik,
        "avg_times": all_times.mean(axis = 1)
    }

