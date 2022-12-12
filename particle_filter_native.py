"""
Native python implementation of the particle filter. Used for comparison against the JAX implementation
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
from jax import lax
from pfjax import particle_resamplers as resampler


def particle_filter_for (model, key, y_meas, theta, n_particles, for_loop=False):
    r"""
    Implementation of the particle filter in Algorithm 1 of Stat 906 project writeup
    """
    n_obs = y_meas.shape[0]
    key, *subkeys = random.split(key, num=n_particles+1)
    x_particles = jnp.zeros((n_particles, *model._n_state))
    logw = jnp.zeros(n_particles)

    # initial particles and weights
    for i, _subkey in enumerate(subkeys):
        init_tmp = model.pf_init(key=_subkey, y_init=y_meas[0], theta=theta)
        x_particles = x_particles.at[i].set(init_tmp[0])
        logw = logw.at[i].set(init_tmp[1])

    # start particle filter: 
    all_particles = jnp.zeros((n_obs, *x_particles.shape))
    all_particles = all_particles.at[0].set(x_particles)
    all_logw = logw
    loglik = jsp.special.logsumexp(logw)
    for t in jnp.arange(1, n_obs):
        key, subkey = random.split(key)

        # resample particles
        resample_out = resampler.resample_multinomial(
            key=subkey,
            x_particles_prev=x_particles,
            logw=logw
        )

        # sample particles for current timepoint
        key, *subkeys = random.split(key, num=n_particles+1)
        x_particles, logw = jax.vmap(
            lambda k, x, y: model.pf_step(key=k, x_prev=x, y_curr=y, theta=theta),
            in_axes=(0, 0, None)
        )(jnp.array(subkeys), resample_out["x_particles"], y_meas[t])
        
        loglik += jsp.special.logsumexp(logw) # log-likelihood calculation
        all_particles = all_particles.at[t].set(x_particles)
        all_logw = all_logw.at[t].set(logw)
    
    return {
        "x_particles": all_particles,
        "logw": all_logw,
        "loglik": loglik - n_obs * jnp.log(n_particles)
    }
    