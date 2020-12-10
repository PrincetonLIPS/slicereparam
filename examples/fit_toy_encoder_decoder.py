from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np 
from jax import jit, grad, vmap
from jax import random
from jax import lax
from jax.ops import index, index_update
from jax.flatten_util import ravel_pytree
from functools import partial

from slicereparam.slicesampler import slicesampler

import matplotlib.pyplot as plt
from tqdm.auto import trange
from jax.scipy.special import expit as sigmoid

# set up randomness
key = random.PRNGKey(1234)

# Set up params
D = 20      # number of latent / data dimensions
N = 1000    # number of data points

# generate data
key, *subkeys = random.split(key, 4)
mu_true = random.normal(subkeys[0], (D,))
z_true = mu_true + random.normal(subkeys[1], (N,D))
x_true = z_true + random.normal(subkeys[2], (N,D))

# init params
key, *subkeys = random.split(key, 4)
mu = random.normal(subkeys[0], (D,))
A  = random.normal(subkeys[1], (D, D))
b  = random.normal(subkeys[2], (D, ))


def gaussian_log_pdf(x, mu, Sigma):
    out = -0.5 * (x - mu).T @ np.linalg.inv(Sigma) @ (x - mu)
    out = out - 0.5 *  np.log(np.linalg.det(Sigma))
    out = out - D / 2.0 * np.log(2.0 * np.pi)
    return out
vmap_gaussian_log_pdf = vmap(gaussian_log_pdf, (0, 0, None))

def _log_pdf(z, params, x):
    A, b, mu = params
    z_mean = A@x + b
    # q(z|x) = N(Ax + b, 2/3 I)
    return gaussian_log_pdf(z, z_mean, 2.0 / 3.0 * np.eye(D))

_params = [A, b, mu]
params, unflatten = ravel_pytree(_params)
log_pdf = jit(lambda z, params, x: _log_pdf(z, unflatten(params), x))
vmapped_log_pdf = jit(vmap(log_pdf, (0,None,0)))

from jax.lax import stop_gradient
@jit
def negative_elbo(zs, xs, params):
    A, b, mu = unflatten(params)
    xhats = zs + 0.0 # mean of x is z
    out = np.sum(-0.5 * (zs - mu[None,:])**2) # p(z) = N(mu, I)
    out = out + np.sum(vmap_gaussian_log_pdf(xs, zs, np.eye(D))) # p(x|z) = N(z, I)
    params = stop_gradient(params)
    out = out - np.sum(vmapped_log_pdf(zs, params, xs)) # entropy term (grad part)
    return -1.0 * out
# gradient of loss with respect to x
loss_grad_xs = jit(grad(negative_elbo))
loss_grad_params = jit(grad(lambda params, zs, xs : negative_elbo(zs, xs, params)))

@jit
def generate_data_idx(key):
    key, subkey = random.split(key)
    data_idx = random.randint(subkey, (S, ), 0, N)
    return data_idx, key

# optimize parameters!
theta = params+0.0
M = theta.shape[0]
losses = [0.0]
thetas = [theta]

# set up iters
S = 128
num_chains = S + 0
Sc = 50
Sl = 20 

# define module
model = slicesampler(theta, log_pdf, D, Sc=Sc, num_chains=num_chains, Sl=Sl)

# learning rate params
a0 = 0.05
gam = 0.0001

num_iters = 1000

pbar = trange(num_iters)
pbar.set_description("Loss: {:.1f}".format(losses[0]))

for i in range(num_iters):

    data_idx, key = generate_data_idx(key)
    ys = x_true[data_idx]

    forwards_out = model.forwards_sample(theta, key, ys=ys)

    # process output
    key = forwards_out[-1]
    xs0 = forwards_out[0] 
    xs = xs0[:,-1:,:].reshape((num_chains, D), order='F') # samples for loss
    dL_dxs = loss_grad_xs(xs, ys, theta)

    # compute gradient
    dL_dtheta = model.compute_gradient_one_sample(theta, dL_dxs, forwards_out)
    dL_dtheta = dL_dtheta + loss_grad_params(theta, xs, ys) / S

    # update loss
    losses.append(-1.0 * negative_elbo(xs, ys, theta))

    # update params
    alpha_t = a0 / (1 + gam * (i+1)) # learning rate 
    theta = theta - dL_dtheta * alpha_t
    model.params = theta
    thetas.append(theta)

    pbar.set_description("Loss: {:.1f}".format(losses[-1]))
    pbar.update()

pbar.close()

thetas_plot = np.array(thetas)
# thetas_reparam_plot = np.array(thetas_reparam)
import seaborn as sns 
sns.set_context("talk")
A_fit, b_fit, mu_fit = unflatten(theta)
mustar = np.mean(x_true, axis=0)
Astar = np.eye(D) / 2 
bstar = mustar / 2.0
plt.figure(figsize=[12,4])
plt.subplot(131)
plt.imshow(np.vstack((mustar, mu_fit, mu)).T, aspect="auto")
plt.xticks([0.0, 1.0, 2.0], ["$\mu^*$", "$\hat{\mu}$", "$\mu_{init}$"])
plt.yticks([])
plt.colorbar()
plt.subplot(132)
plt.imshow(np.vstack((bstar, b_fit, b)).T, aspect="auto")
plt.xticks([0.0, 1.0, 2.0], ["$b^*$", "$\hat{b}$", "$b_{init}$"])
plt.yticks([])
plt.colorbar()
plt.subplot(133)
plt.imshow(np.vstack((Astar, A_fit, A)).T, aspect="auto")
plt.xticks([10.0, 30.0, 50.0], ["$A^*$", "$\hat{A}$", "$A_{init}$"])
plt.yticks([])
plt.colorbar()
plt.tight_layout()
