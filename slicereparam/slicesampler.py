import jax.numpy as np 
from jax import jit, grad, vmap
from jax import random
from jax import lax
from jax.ops import index, index_update
from jax.flatten_util import ravel_pytree
from functools import partial

import slicereparam.rootfinder as rootfinder

class slicesampler(object):
    """

    """
    def __init__(self, params, log_pdf, loss, D, Sc=1, num_chains=1, **kwargs):

        params, unflatten = ravel_pytree(_params)
        log_pdf = jit(lambda x, params : log_pdf(x, unflatten(params)))
        vmapped_log_pdf = jit(vmap(log_pdf, (0,None)))

        # compute necessary gradients
        def log_pdf_theta(theta, x):    return log_pdf(x, theta)
        def log_pdf_x(x, theta):        return log_pdf(x, theta)
        def log_pdf_ad(x, theta, a, d): return log_pdf(x + a * d, theta)
        grad_x = jit(grad(log_pdf_x))
        grad_theta = jit(grad(log_pdf_theta))
        grad_x_ad = jit(grad(log_pdf_ad))

        # grad log normalizer of posterior
        vmapped_grad_theta = jit(vmap(grad_theta, (None,0)))

        # total_loss = jit(lambda x, params : loss(x, unflatten(params)))
        total_loss = jit(lambda x, params : loss(x, params))

        loss_grad_xs = jit(grad(total_loss))
        # loss_grad_params = jit(grad(lambda params, x, y : total_loss(x, y, params)))
        loss_grad_params = jit(grad(lambda params, x : total_loss(x, params)))

    def forwards_step(self, x, theta, u1, u2, d):#, aL, bR):
        func = lambda alpha : self.log_pdf(x + alpha * d, theta) - self.log_pdf(x, theta) - np.log(u1) # root
        aL, bR = choose_start(func)
        z_L, z_R = dual_bisect_method(func, aL=aL, bL=-1e-10, aR=1e-10, bR=bR)
        x_L = x + d*z_L
        x_R = x + d*z_R
        x = (1 - u2) * x_L + u2 * x_R
        alphas = np.array([z_L, z_R])
        return x, x_L, x_R, alphas

    def vmapped_forwards_step(self, x, theta, u1, u2, d):
        return (vmap(self.forwards_step, (0,None,0,0,0)))

    def forwards(self, S, theta, x, us, ds):
        xs = np.zeros((S+1, num_chains, D))
        xs = index_update(xs, index[0, :, :], x)
        xLs = np.zeros((S, num_chains, D))
        xRs = np.zeros((S, num_chains, D))
        alphas = np.zeros((S, num_chains, 2))
        init_val = [xs, xLs, xRs, alphas, x]

        def body_fun(i, val):
            xs, xLs, xRs, alphas, x = val 
            x, x_L, x_R, alpha = self.vmapped_forwards_step(x, theta, us[i,:,0], us[i,:,1], ds[i])
            xs = index_update(xs, index[i+1, :, :], x)
            xLs = index_update(xLs, index[i, :, :], x_L)
            xRs = index_update(xRs, index[i, :, :], x_R)
            alphas = index_update(alphas, index[i, :, :], alpha)
            val = [xs, xLs, xRs, alphas, x]
            return val

        xs, xLs, xRs, alphas, x = lax.fori_loop(0, S, body_fun, init_val)
        return xs, xLs, xRs, alphas

    # set up randomness
    def generate_randomness(self, key):
        key, *subkeys = random.split(key, 4)
        us = random.uniform(subkeys[0], (self.Sc,self.num_chains,2))
        ds = random.normal(subkeys[1], (self.Sc*self.num_chains,self.D))
        ds_norm = ds / np.sqrt(np.sum(ds**2, axis=1))[:,None]
        ds_norm = ds_norm.reshape((self.Sc, self.num_chains, self.D))
        x0 = random.normal(subkeys[2], (self.num_chains, self.D))
        return us, ds_norm, x0, key

    @partial(jit, static_argnums=(0))
    def forwards_sample(self, theta, key):
        us, norm_ds, x0, key = self.generate_randomness(key)
        key, subkey = random.split(key)
        x0 = theta[:D] + np.sqrt(np.exp(theta[D:])) * random.normal(subkey, (num_chains, D))
        xs0, xLs, xRs, alphas = self.forwards(Sc, theta, x0, us, norm_ds)
        return xs0, us, norm_ds, xLs, xRs, alphas, key
