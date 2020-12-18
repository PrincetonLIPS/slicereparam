from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np 
from jax import jit, grad, vmap
from jax import random
from jax import lax
from jax.lax import stop_gradient
from jax.ops import index, index_update
from jax.flatten_util import ravel_pytree
from functools import partial

from slicereparam.functional import setup_slice_sampler

import matplotlib.pyplot as plt
from tqdm.auto import trange
from jax.scipy.special import expit as sigmoid

# set up randomness
key = random.PRNGKey(1234)

# Set up params
D = 5   # number of dimensions

# initialize params
scale = 0.1
key, *subkeys = random.split(key, 3)
# mean, log variance of diagonal Gaussian
_params = [scale * random.normal(subkeys[0], (D, )), 
           scale * random.normal(subkeys[1], (D, ))]

# log pdf function (up to additive constant)
def _log_pdf(x, params):
    mu = params[0]
    sigma_diag = np.exp(params[1])
    return np.sum(-0.5 * (x - mu) **2 / sigma_diag)
params, unflatten = ravel_pytree(_params)
log_pdf = jit(lambda x, params : _log_pdf(x, unflatten(params)))
vmapped_log_pdf = jit(vmap(log_pdf, (0,None)))

# target diagonal Gaussian 
key, *subkeys = random.split(key, 3)
xstar = random.normal(subkeys[0], (D,) )
true_var = np.exp(-0.5 + 0.5 * random.normal(subkeys[1], (D,)))
Sigma = np.diag(true_var)
@jit
def gaussian_log_pdf(x, mu, Sigma):
    out = -0.5 * (x - mu).T @ np.linalg.inv(Sigma) @ (x - mu)
    out = out - 0.5 *  np.log(np.linalg.det(Sigma))
    out = out - D / 2.0 * np.log(2.0 * np.pi)
    return out
vmap_gaussian_log_pdf = vmap(gaussian_log_pdf, (0, None, None))

num_chains = 1
S = 10
slice_sample = setup_slice_sampler(log_pdf, D, S, num_chains=num_chains)

# slice reparam loss and gradient
@jit
def loss_slice(params, x0, key):
    xs_all = slice_sample(params, x0, key)
    xs = xs_all[:, -1, :]
    params = stop_gradient(params)
    loss = -1.0 * np.mean(vmap_gaussian_log_pdf(xs, xstar, Sigma)) #- np.sum(0.5 * params[1])
    loss = loss + np.mean(vmapped_log_pdf(xs, params)) # entropy term (grad part)
    return loss
grad_slice = jit(grad(loss_slice))

# reparameterization gradient
def _loss_reparam(params, ds):
    xs = params[0] + np.sqrt(np.exp(params[1])) * ds
    loss = np.mean(vmap_gaussian_log_pdf(xs, xstar, Sigma))
    loss = loss - np.mean(vmap_gaussian_log_pdf(xs, params[0], np.diag(np.exp(params[1]))))
    return -1.0 * loss
loss_reparam = jit(lambda params, ds : _loss_reparam(unflatten(params), ds))
grad_loss_reparam = jit(grad(loss_reparam))

# optimize parameters!
theta = params+0.0
M = theta.shape[0]
losses = [0.0]
thetas = [theta]
theta_reparam = theta + 0.0
thetas_reparam = [theta]

# learning rate params
a0 = 0.1
a0_reparam = 0.1
gam = 0.01

num_iters=1000

pbar = trange(num_iters)
pbar.set_description("Loss: {:.1f}".format(losses[0]))
for i in range(num_iters):

    key, *subkeys = random.split(key, 3)
    x0 = random.normal(subkeys[0], (num_chains, D))
    dL_dtheta = grad_slice(theta, x0, subkeys[1])
    # TODO - combine grad and val in one function
    losses.append(loss_slice(theta, x0, subkeys[1]))

    # update params
    alpha_t = a0 / (1 + gam * (i+1)) # learning rate 
    theta = theta - dL_dtheta * alpha_t
    thetas.append(theta)

    # reparam trick 
    key, subkey = random.split(key)
    ds = random.normal(subkey, (S, D))
    dL_dtheta_reparam = grad_loss_reparam(theta_reparam, ds)
    alpha_t = a0_reparam / (1 + gam * (i+1)) # learning rate 
    theta_reparam = theta_reparam - dL_dtheta_reparam * alpha_t
    thetas_reparam.append(theta_reparam)

    pbar.set_description("Loss: {:.1f}".format(losses[-1]))
    pbar.update()

pbar.close()

thetas_plot = np.array(thetas)
thetas_reparam_plot = np.array(thetas_reparam)

plt.figure(figsize=[8,4])
plt.subplot(121)
for i in range(D):
    plt.axhline(xstar[i],color='k', label="true" if i == 0 else None)
    plt.plot(thetas_reparam_plot[:,i], 'b', label="standard reparam" if i ==0 else None, alpha=0.8)
    plt.plot(thetas_plot[:,i], 'r', label="slice reparam" if i ==0 else None, alpha=0.8)
plt.xlabel("iteration")
plt.ylabel("$\mu$")

plt.subplot(122)
for i in range(D):
    plt.axhline(true_var[i], color='k', label="true" if i == 0 else None)
    plt.plot(np.exp(thetas_reparam_plot[:,i+D]), 'b', label="standard reparam" if i ==0 else None, alpha=0.8)
    plt.plot(np.exp(thetas_plot[:,i+D]), 'r', label="slice reparam" if i ==0 else None, alpha=0.8)
plt.legend()
plt.xlabel("iteration")
plt.ylabel("$\sigma^2$")