from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp 
from jax import jit, grad, vmap
from jax import random
from jax import lax
from jax.ops import index, index_update
from jax.flatten_util import ravel_pytree
from functools import partial

import matplotlib.pyplot as plt
from jax.scipy.special import expit as sigmoid
from jax.scipy.special import logsumexp

from slicereparam.reflective import setup_reflective 
from slicereparam.functional import setup_slice_sampler

import time

key = random.PRNGKey(123)
D = 2
# initialize params
scale = 0.1
key, *subkeys = random.split(key, 3)
# mean, log variance of diagonal Gaussian
_mu = scale * random.normal(subkeys[0], (D, ))
_Sigma = jnp.array([[1.0, 0.9], [0.9, 1.0]])
_params = [_mu, _Sigma]

# log pdf function (up to additive constant)
@jit
def _log_pdf(x, params):
    mu, Sigma = params 
    out = -0.5 * (x - mu).T @ jnp.linalg.inv(Sigma) @ (x - mu)
    out = out - 0.5 *  jnp.log(jnp.linalg.det(Sigma))
    out = out - D / 2.0 * jnp.log(2.0 * jnp.pi)
    return out
params, unflatten = ravel_pytree(_params)
log_pdf = jit(lambda x, params : _log_pdf(x, unflatten(params)))
vmapped_log_pdf = jit(vmap(log_pdf, (0,None)))

S = 20 # number of samples
num_chains = 1 # number of chains
w = 20.0 # step size
reset_iters = 1 # resample
reflective_slice_sampler = setup_reflective(log_pdf, D, S, num_chains, w, reset_iters)
slice_sampler = setup_slice_sampler(log_pdf, D, S, num_chains=num_chains)

x0 = jnp.array([0.1, 0.0])
x0 = x0[None, :]

key, subkey = random.split(key)
out = reflective_slice_sampler(params, x0, subkey)
xs = out[0]
key, subkey = random.split(key)
out2 = slice_sampler(params, x0, subkey)
xs2 = out2[0]

plt.figure()
plt.plot(xs[:, 0], xs[:, 1])
plt.plot(xs2[:, 0], xs2[:, 1])

# compute KL divergence numerically
dz=0.025
z1_range = jnp.arange(-5., 5., dz)
z2_range = jnp.arange(-5.0, 5, dz)
z1s, z2s = jnp.meshgrid(z1_range, z2_range)
@partial(jnp.vectorize, signature='(),()->()')
def q_logz_gauss(z1, z2):
    return log_pdf(jnp.array([z1,z2]), params)
q_ys_gauss = jnp.exp(q_logz_gauss(z1s, z2s))


plt.figure(figsize=[8,4])
plt.subplot(121)
plt.contourf(z1_range, z2_range, q_ys_gauss)#, vmin=0.0,vmax=0.45)
plt.plot(xs[:, 0], xs[:, 1], 'r', '-.', label="reflective")
plt.plot(xs2[:, 0], xs2[:, 1], 'b', '-.', label="random")
# plt.title("target")
plt.xlim([-3,3])
plt.ylim([-3,3])

plt.subplot(122)
plt.contourf(z1_range, z2_range, q_ys_gauss)#, vmin=0.0,vmax=0.45)
plt.plot(xs[:, 0], xs[:, 1], 'r.', label="reflective")
plt.plot(xs2[:, 0], xs2[:, 1], 'b.', label="random")
# plt.title("target")
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend()