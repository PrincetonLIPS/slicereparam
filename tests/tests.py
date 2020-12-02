from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp 
from jax import jit, grad, vmap
from jax import random
from jax.flatten_util import ravel_pytree
from jax.scipy.special import logsumexp

from slicereparam.slicesampler import slicesampler

def test_grad_diagonal_gaussian_KL():

    # set up randomness
    key = random.PRNGKey(131313)

    # Set up params
    D = 5   # number of dimensions
    scale = 0.1
    key, *subkeys = random.split(key, 3)
    _params = [scale * random.normal(subkeys[0], (D, )), scale * random.normal(subkeys[1], (D, ))]

    def _log_pdf(x, params):
        mu = params[0]
        sigma_diag = jnp.exp(params[1])
        return jnp.sum(-0.5 * (x - mu) **2 / sigma_diag)

    params, unflatten = ravel_pytree(_params)
    log_pdf = jit(lambda x, params : _log_pdf(x, unflatten(params)))
    vmapped_log_pdf = jit(vmap(log_pdf, (0,None)))

    xstar = jnp.zeros(D)
    Sigma = jnp.eye(D)

    def gaussian_log_pdf(x, mu, Sigma):
        out = -0.5 * (x - mu).T @ jnp.linalg.inv(Sigma) @ (x - mu)
        out = out - 0.5 *  jnp.log(jnp.linalg.det(Sigma))
        out = out - D / 2.0 * jnp.log(2.0 * jnp.pi)
        return out

    vmap_gaussian_log_pdf = vmap(gaussian_log_pdf, (0, None, None))

    def loss(xs, params):
        loss = -1.0 * jnp.sum(vmap_gaussian_log_pdf(xs, xstar, Sigma)) 
        loss = loss + jnp.sum(vmapped_log_pdf(xs, params)) 
        return loss

    total_loss = jit(lambda x, params : loss(x, params))

    num_chains = 10000
    Sc = 100
    model = slicesampler(
        params, log_pdf, D, total_loss=total_loss, Sc=Sc, num_chains=num_chains)
    dL_dtheta, loss, key = model.estimate_gradient(params, key)

    def true_loss(params):
        mu, log_sigsqr = params
        return 0.5 * jnp.sum(jnp.exp(log_sigsqr) + mu**2 + 1.0 - log_sigsqr)

    true_grad = grad(lambda params : true_loss(unflatten(params)))

    assert jnp.linalg.norm(dL_dtheta - true_grad(params)) < 1e-2

def test_sampler_cdf():

    var1, var2, var3 = 2.0, 1.0, 1.5
    w1, w2, w3 = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    def _log_pdf(x, params):
        mu1, mu2, mu3 = params[:3]
        log1 = -0.5 * (x - mu1)**2 / var1 - 0.5 * jnp.sqrt(2.0 * jnp.pi * var1) + jnp.log(w1)
        log2 = -0.5 * (x - mu2)**2 / var2 - 0.5 * jnp.sqrt(2.0 * jnp.pi * var2) + jnp.log(w2)
        log3 = -0.5 * (x - mu3)**2 / var3 - 0.5 * jnp.sqrt(2.0 * jnp.pi * var3) + jnp.log(w3)
        return jnp.sum(logsumexp(jnp.array([log1,log2,log3]),axis=0))
    _params = [-4.0, 0.0, 4.0]
    params, unflatten = ravel_pytree(_params)
    log_pdf = jit(lambda x, params : _log_pdf(x, unflatten(params)))

    D = 1
    Sc = 5000
    num_chains = 100
    key = random.PRNGKey(5)
    model = slicesampler(
        params, log_pdf, D, total_loss=None, Sc=Sc, num_chains=num_chains)
    out = model.forwards_sample(key)
    xs0 = out[0]
    xs = xs0[1:].reshape(num_chains * Sc, D)

    dx = 0.01
    x_range = jnp.arange(-12,12,dx)
    pdf = jnp.array([jnp.exp(log_pdf(x, params)) for x in x_range])
    numerical_cdf = jnp.cumsum(pdf / jnp.sum(pdf))
    empirical_cdf = jnp.array([jnp.sum(xs < x) for x in x_range]) / (Sc * num_chains)

    assert jnp.linalg.norm(numerical_cdf - empirical_cdf) < 0.1


if __name__ == "__main__":
    test_grad_diagonal_gaussian_KL()
    test_sampler_cdf()