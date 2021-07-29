from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, grad, vmap
from jax import random
from jax import lax
from jax import custom_vjp
from jax.ops import index, index_update
from jax.flatten_util import ravel_pytree
from functools import partial

from slicereparam.rootfinder import dual_bisect_method, \
    bisect_method, choose_start, single_choose_start

from inspect import signature
import warnings 

def setup_reflective(log_pdf, D, S, num_chains=1, w=1.0, reset_iters=10):
    # w is step size
    # resample momentum every `resample` steps

    # set up for backwards pass
    # compute necessary gradients
    # TODO - modify code so log_pdf is always called in same order (fix the theta switch, just take grad differently).
    def log_pdf_theta(theta, x):    return log_pdf(x, theta)
    def log_pdf_x(x, theta):        return log_pdf(x, theta)
    def log_pdf_ad(x, theta, a, d): return log_pdf(x + a * d, theta)
    grad_x = jit(grad(log_pdf_x))
    grad_theta = jit(grad(log_pdf_theta))
    grad_x_ad = jit(grad(log_pdf_ad))

    def forwards_step(x, theta, u1, p, w, maxiter=100):
        wp = w + 0.0 # wprime is current step size
        wt = 0.0 # wt tracks accumulated distance
        xp = x + 0.0 # xprime tracks current state
        # init_val = [xp, p, wp, wt]
        init_val = [xp, p, wp]

        # TODO: also keep track of list of x boundary hits, h's, w's? basically, the path? 

        def cond_fun(val):
            # run while w < alpha
            # xp, p, wp, wt = val 
            # return wt < w 
            xp, p, wp = val 
            return wp > 0

        def update_fun(val):
            # xp, p, wp, wt, alpha = val
            xp, p, wp, alpha = val
            xp = xp + wp * p 
            # wt = wt + wp 
            # return [xp, p, wp, wt, alpha]
            wp = 0.0
            return [xp, p, wp, alpha]

        def reflect_fun(val):
            # xp, p, wp, wt, alpha = val
            xp, p, wp, alpha = val
            xp = xp + alpha * p 
            h = grad_x(xp, theta)
            # p = p - 2. * h * jnp.dot(p, h) / (jnp.dot(h, h)**2)
            p = p - 2. * h * jnp.dot(p, h) / (jnp.dot(h, h))
            wp = wp - alpha 
            # wt = wt + alpha 
            # return [xp, p, wp, wt, alpha]
            return [xp, p, wp, alpha]

        def body_fun(val):
            # xp, p, wp, wt = val # unpack
            xp, p, wp = val # unpack
            # init root finding function
            # we keep x in second term because it defines slice height
            func = lambda alpha : log_pdf(xp + alpha * p, theta) - log_pdf(x, theta) - jnp.log(u1) # root
            # choose start
            b = single_choose_start(func)
            # bisect (single bisection), bracket is [a0, b]
            alpha = bisect_method(func, b=b, a=1e-10)

            # update
            # cond_val = [xp, p, wp, wt, alpha]
            cond_val = [xp, p, wp, alpha]
            cond_val = lax.cond(wp < alpha, update_fun, reflect_fun, cond_val)
            # [xp, p, wp, wt, alpha] = cond_val
            [xp, p, wp, alpha] = cond_val

            # return [xp, p, wp, wt]
            return [xp, p, wp]

        val = lax.while_loop(cond_fun, body_fun, init_val)
        # unpack val
        # [xp, p, wp, wt] = val
        [xp, p, wp] = val

        # return xp, p, wp, wt
        # return current location and momentum 
        return xp, p

    @jit 
    def forwards_sample(theta, x0, key):
        # forwards_step(x, theta, u1, p, w, maxiter=100):
        # here, run forwards step multiple times
        # each time, sample new value for p?
        # but keep w the same. 

        # generate randomness - u1s, ds
        key, *subkeys = random.split(key, 3)
        us = random.uniform(subkeys[0], (num_chains, S))
        # TODO - could decrease number of d's sampled here. 
        ds_unnorm = random.normal(subkeys[1], (S * num_chains, D))
        ds = ds_unnorm / jnp.sqrt(jnp.sum(ds_unnorm**2, axis=1))[:,None]
        ds = ds.reshape((num_chains, S, D))

        # initialize variables
        xs = jnp.zeros((num_chains, S+1, D))
        xs = index_update(xs, index[:, 0, :], x0)
        p = ds[:, 0, :] # initial p
        init_val = [xs, p, x0]

        # def reset_momentum(i, reset_iters):
        #     """Returns true if reset momentum.
        #     Reset_iters is number of iters before resetting momentum."""
        #     if jnp.mod(i, reset_iters) == 0:
        #         return True 
        #     else:
        #         return False

        def body_fun(i, val):
            xs, p, x = val 
            # xp, p = forwards_step(x, theta, u1, p, w, maxiter=100)
            # need a new u1 at each iteration, but can keep around p.
            x, p = vmap(forwards_step, (0,None,0,0,None))(x, theta, us[:,i], p, w)
            xs = index_update(xs, index[:, i+1, :], x)
            # reset momentum every reset_iters 
            p = lax.cond(jnp.mod(i+1, reset_iters) == 0, 
                         lambda p : ds[:, i, :],
                         lambda p : p,
                         p)
            val = [xs, p, x]
            return val
        
        xs, p, x = lax.fori_loop(0, S, body_fun, init_val)
        
        return xs

    return forwards_sample 