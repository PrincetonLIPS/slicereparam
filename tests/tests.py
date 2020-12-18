from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp 
from jax import jit, grad, vmap
from jax import random
from jax.ops import index, index_update
from jax.flatten_util import ravel_pytree
from jax.scipy.special import logsumexp

from slicereparam.functional import setup_slice_sampler
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
    model = slicesampler(params, log_pdf, D, Sc=Sc, num_chains=num_chains)
    out = model.forwards_sample(params, key)
    xs0 = out[0]
    xs = xs0[:,1:,:].reshape(num_chains * Sc, D)

    dx = 0.01
    x_range = jnp.arange(-12,12,dx)
    pdf = jnp.array([jnp.exp(log_pdf(x, params)) for x in x_range])
    numerical_cdf = jnp.cumsum(pdf / jnp.sum(pdf))
    empirical_cdf = jnp.array([jnp.sum(xs < x) for x in x_range]) / (Sc * num_chains)

    assert jnp.linalg.norm(numerical_cdf - empirical_cdf) < 0.1


def test_finite_difference():

    # write a test to estimate gradient via slice sampling and finite diff
    # make sure close 
    # set up randomness
    key = random.PRNGKey(1234)

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

    def _total_loss(xs, params):
        loss = jnp.sum(xs**2)
        return loss
    total_loss = jit(lambda x, params : _total_loss(x, params))
    loss_grad_xs = jit(grad(total_loss))

    # run test over 1, >1 number of MCMC chains and 1, >1 number of samples
    num_chains_vals = [1, 2]
    Sc_vals = [1, 5]

    for num_chains, Sc in zip(num_chains_vals, Sc_vals):

        model = slicesampler(params, log_pdf, D, Sc=Sc, num_chains=num_chains)
        forwards_out = model.forwards_sample(params, key)

        xs = forwards_out[0][:, 1:, :]
        dL_dxs = loss_grad_xs(xs, params)

        dL_dtheta = model.compute_gradient(params, dL_dxs, forwards_out)

        # compute gradient via finite differences
        dx = 1e-3
        M = params.shape[0]
        dthetas = [jnp.zeros_like(params) for nc in range(num_chains)]
        for m, v in enumerate(jnp.eye(M)):
            params1 = params - dx * v
            params2 = params + dx * v
            forwards_out1 = model.forwards_sample(params1, key)
            model.params = params2
            forwards_out2 = model.forwards_sample(params2, key)
            # xs1 = forwards_out1[0][1:].reshape((num_chains, Sc, D), order='F')
            # xs2 = forwards_out2[0][1:].reshape((num_chains, Sc, D), order='F')
            xs1 = forwards_out1[0][:, 1:, :]
            xs2 = forwards_out2[0][:, 1:, :]
            for nc in range(num_chains):
                loss1 = total_loss(xs1[nc], params1)
                loss2 = total_loss(xs2[nc], params2) 
                dthetas[nc] = dthetas[nc] + (loss2 - loss1) / (2.0 * dx) * v 

        dthetas = jnp.mean(jnp.asarray(dthetas), axis=0)
        assert jnp.linalg.norm(dL_dtheta - dthetas) < 1e-2

        key = forwards_out[-1]

# def test_root_finder():
#     return 

def test_custom_vjp_finite_difference():

    # write a test to estimate gradient via slice sampling and finite diff
    # make sure close 
    # set up randomness
    key = random.PRNGKey(123)

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

    # run test over 1, >1 number of MCMC chains and 1, >1 number of samples
    num_chains_vals = [1, 1, 2, 2]
    S_vals = [1, 5, 1, 5]

    for S, num_chains in zip(S_vals, num_chains_vals):

        # slice_sample = setup_slice_sampler(log_pdf, params, D, S, num_chains=num_chains)
        slice_sample = setup_slice_sampler(log_pdf, D, S, num_chains=num_chains)

        key, *subkeys = random.split(key, 3)
        x0 = random.normal(subkeys[0], (num_chains, D))
        # out = slice_sample(subkeys[1], params, x0)
        out = slice_sample(params, x0, subkeys[1])

        def loss(xs):
            return jnp.mean(xs**2)

        def compute_loss(params, x0, key):
            xs = slice_sample(params, x0, key)
            return loss(xs)

        grad_loss = jit(grad(compute_loss))

        key, *subkeys = random.split(key, 3)
        x0 = random.normal(subkeys[0], (num_chains, D))
        grad_params_ad = grad_loss(params, x0, subkeys[1])

        # compute gradient via finite differences
        dx = 1e-3
        M = params.shape[0]
        dthetas = [jnp.zeros_like(params) for nc in range(num_chains)]
        for m, v in enumerate(jnp.eye(M)):
            params1 = params - dx * v
            params2 = params + dx * v
            xs1 = slice_sample(params1, x0, subkeys[1])
            xs2 = slice_sample(params2, x0, subkeys[1])
            for nc in range(num_chains):
                loss1 = loss(xs1[nc])
                loss2 = loss(xs2[nc])
                dthetas[nc] = dthetas[nc] + (loss2 - loss1) / (2.0 * dx) * v 

        grad_params_fd = jnp.mean(jnp.asarray(dthetas), axis=0)

        grad_x0 = jit(grad(compute_loss, argnums=1))
        grad_x0_ad = grad_x0(params, x0, subkeys[1])
        dx = 1e-3
        dxs = [jnp.zeros(D) for nc in range(num_chains)]
        for nc in range(num_chains):
            for m, v in enumerate(jnp.eye(D)):
                x01 = x0[nc] - dx * v
                x02 = x0[nc] + dx * v
                x01 = index_update(x0, index[nc, :], x01)
                x02 = index_update(x0, index[nc, :], x02)
                xs1 = slice_sample(params, x01, subkeys[1])
                xs2 = slice_sample(params, x02, subkeys[1])
                loss1 = loss(xs1)
                loss2 = loss(xs2)
                dxs[nc] = dxs[nc] + (loss2 - loss1) / (2.0 * dx) * v 
        grad_x0_fd = jnp.asarray(dxs)

        assert jnp.linalg.norm(grad_params_ad - grad_params_fd) < 1e-3
        assert jnp.linalg.norm(grad_x0_ad - grad_x0_fd) < 1e-3


if __name__ == "__main__":
    test_grad_diagonal_gaussian_KL()
    test_sampler_cdf()
    test_finite_difference()
    test_custom_vjp_finite_difference()