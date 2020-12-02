from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np 
from jax import jit, grad, vmap
from jax import random
from jax.flatten_util import ravel_pytree

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
        sigma_diag = np.exp(params[1])
        return np.sum(-0.5 * (x - mu) **2 / sigma_diag)

    params, unflatten = ravel_pytree(_params)
    log_pdf = jit(lambda x, params : _log_pdf(x, unflatten(params)))
    vmapped_log_pdf = jit(vmap(log_pdf, (0,None)))

    xstar = np.zeros(D)
    Sigma = np.eye(D)

    def gaussian_log_pdf(x, mu, Sigma):
        out = -0.5 * (x - mu).T @ np.linalg.inv(Sigma) @ (x - mu)
        out = out - 0.5 *  np.log(np.linalg.det(Sigma))
        out = out - D / 2.0 * np.log(2.0 * np.pi)
        return out

    vmap_gaussian_log_pdf = vmap(gaussian_log_pdf, (0, None, None))

    def loss(xs, params):
        loss = -1.0 * np.sum(vmap_gaussian_log_pdf(xs, xstar, Sigma)) 
        loss = loss + np.sum(vmapped_log_pdf(xs, params)) 
        return loss

    total_loss = jit(lambda x, params : loss(x, params))

    num_chains = 10000
    Sc = 100
    model = slicesampler(params, log_pdf, D, total_loss, Sc=Sc, num_chains=num_chains)
    dL_dtheta, loss, key = model.estimate_gradient(params, key)

    def true_loss(params):
        mu, log_sigsqr = params
        return 0.5 * np.sum(np.exp(log_sigsqr) + mu**2 + 1.0 - log_sigsqr)

    true_grad = grad(lambda params : true_loss(unflatten(params)))

    assert np.linalg.norm(dL_dtheta - true_grad(params)) < 1e-2

if __name__ == "__main__":
    test_grad_diagonal_gaussian_KL()
