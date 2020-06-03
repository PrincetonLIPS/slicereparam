from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np 
from jax import jit, grad
from jax import random
from jax import lax
from jax.flatten_util import ravel_pytree
from functools import partial

from scipy.optimize import root_scalar, brentq
import matplotlib.pyplot as plt
from tqdm.auto import trange
from jax.scipy.special import expit as sigmoid
from jax.scipy.special import logsumexp

@jit
def f_alpha(alpha, x, d, theta, u1):
    return log_pdf(x + alpha * d, theta) - log_pdf(x, theta) - np.log(u1)

@jit
def fa(x, alpha, d, theta, u1):
    return log_pdf(x + alpha * d, theta) - log_pdf(x, theta) - np.log(u1)

@jit
def dual_bisect_method(
    x, d, theta, u1,
    aL=-1e5, bL=-1e-5, aR=1e-5, bR=1e5,
    tol=1e-8, maxiter=100):

    i = maxiter-1.0
    init_val = [aL, bL, aR, bR, i]

    def cond_fun(val):
        aL, bL, aR, bR, i = val
        # return np.maximum(bL-aL, bR-aR) > 2.0 * tol
        # return np.sum((b-a) / 2.0) > tol
        return np.sum(bL-aL) + np.sum(bR-aR) + 100 * np.minimum(i, 0.0) > tol

    def body_fun(val):

        aL, bL, aR, bR, i = val
        cL = (aL+bL)/2.0
        cR = (aR+bR)/2.0

        # alphas = np.array([aL, bL, cL, aR, bR, cR])
        # sign_aL, sign_bL, sign_cL, sign_aR, sign_bR, sign_cR = np.sign(fa_batched(x, alphas, d, theta, u1))

        # L
        sign_cL = np.sign(fa(x, cL, d, theta, u1))
        sign_aL = np.sign(fa(x, aL, d, theta, u1))
        sign_bL = np.sign(fa(x, bL, d, theta, u1))
        aL = np.sum(cL * np.maximum( sign_cL * sign_aL, 0.0) + \
            aL * np.maximum( -1.0 * sign_cL * sign_aL, 0.0))
        bL = np.sum(cL * np.maximum( sign_cL * sign_bL, 0.0) + \
            bL * np.maximum( -1.0 * sign_cL * sign_bL, 0.0))

        # R
        sign_cR = np.sign(fa(x, cR, d, theta, u1))
        sign_aR = np.sign(fa(x, aR, d, theta, u1))
        sign_bR = np.sign(fa(x, bR, d, theta, u1))
        aR = np.sum(cR * np.maximum( sign_cR * sign_aR, 0.0) + \
            aR * np.maximum( -1.0 * sign_cR * sign_aR, 0.0))
        bR = np.sum(cR * np.maximum( sign_cR * sign_bR, 0.0) + \
            bR * np.maximum( -1.0 * sign_cR * sign_bR, 0.0))

        i = i + 1
        val = [aL, bL, aR, bR, i]

        return val

    val = lax.while_loop(cond_fun, body_fun, init_val)
    aL, bL, aR, bR, i = val 
    cL = (aL+bL)/2.0
    cR = (aR+bR)/2.0
    return [cL, cR]

a_grid = np.concatenate((np.logspace(-3,1, 25), np.array([25.0])))
# @jit
@jit
def forwards_step(x, theta, u1, u2, d, aL, bR):
    # z_L = bisect_method(x, d, theta, u1, a=-25.0, b=-1e-10)
    # z_R = bisect_method(x, d, theta, u1, a=1e-10, b=25.0)
    z_L, z_R = dual_bisect_method(x, d, theta, u1, aL=aL, bL=-1e-10, aR=1e-10, bR=bR)
    x_L = x + d*z_L
    x_R = x + d*z_R
    x = (1 - u2) * x_L + u2 * x_R
    alphas = np.array([z_L, z_R])
    return x, x_L, x_R, alphas

# @jit
def forwards(S, theta, x, us, ds):
    xs = [x]
    xLs = []
    xRs = []
    alphas = []
    for s in range(S):
        aL=a_grid[np.where(fa(x, -a_grid, ds[s], theta, us[s,0])<0)[0][0]]*-1.0
        bR=a_grid[np.where(fa(x, a_grid, ds[s], theta, us[s,0])<0)[0][0]]
        x, x_L, x_R, alpha = forwards_step(x, theta, us[s,0], us[s,1], ds[s], aL, bR)
        xs.append(x)
        xLs.append(x_L)
        xRs.append(x_R)
        alphas.append(alpha)
    return np.array(xs), np.array(xLs), np.array(xRs), np.array(alphas)

def _log_pdf(x, params):

    mu1 = params[0]
    mu2 = params[1]

    log1 = -0.5 * (x - mu1)**2 - 0.5 * np.sqrt(2.0 * np.pi)
    log2 = -0.5 * (x - mu2)**2 - 0.5 * np.sqrt(2.0 * np.pi)
    return logsumexp(np.array([log1,log2]),axis=0)

# compute necessary gradients
def log_pdf_theta(theta, x):    return log_pdf(x, theta)
def log_pdf_x(x, theta):        return log_pdf(x, theta)
def log_pdf_ad(x, theta, a, d): return log_pdf(x + a * d, theta)
grad_x = jit(grad(log_pdf_x))
grad_theta = jit(grad(log_pdf_theta))
grad_x_ad = jit(grad(log_pdf_ad))

_params = [-4.0, 4.0]
params, unflatten = ravel_pytree(_params)
log_pdf = jit(lambda x, params : _log_pdf(x, unflatten(params)))

D = 1
key = random.PRNGKey(3)

S = 10000 # number of samples
key, *subkeys = random.split(key, 4)
us = random.uniform(subkeys[0], (S,2))
ds = random.normal(subkeys[1], (S,D))
ds_norm = np.array([d / np.linalg.norm(d) for d in ds])
x = 0.1 * random.normal(subkeys[2], (D,)) # initial x 

# run forward pass
xs, xLs, xRs, alphas = forwards(S, params, x, us, ds_norm)

x_range = np.arange(-10,10,0.01)
g1 = 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * (x_range - _params[0])**2)
g2 = 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * (x_range - _params[1])**2)
z1 = 0.5 
plt.figure(figsize=[8,4])
plt.plot(x_range, z1 * g1 + (1-z1) * g2)
plt.hist(xs[:,0], 100, density=True)

S=10
xs2_all = np.array([])
for i in range(12):
    key, *subkeys = random.split(key, 4)
    us = random.uniform(subkeys[0], (S,2))
    ds = random.normal(subkeys[1], (S,D))
    ds_norm = np.array([d / np.linalg.norm(d) for d in ds])
    x = 0.1 * random.normal(subkeys[2], (D,)) # initial x 
    xs2, xLs2, xRs2, alphas2 = forwards(S, params, x, us, ds_norm)
    xs2_all = np.append(xs2_all, xs2[1:][:,0])
xs2_all=np.array(xs2_all)

x_range = np.arange(-10,10,0.01)
g1 = 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * (x_range - _params[0])**2)
g2 = 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * (x_range - _params[1])**2)
z1 = 0.5 
plt.figure(figsize=[8,4])
plt.plot(x_range, z1 * g1 + (1-z1) * g2)
plt.hist(xs2_all, 40, density=True)

# def forwards2(S, theta, x, us, ds):
#     xs = [x]
#     xLs = []
#     xRs = []
#     alphas = []

#     for s in range(S):

#         # import ipdb; ipdb.set_trace()
#         u1 = us[s,0]
#         u2 = us[s,1]
#         d = ds[s]

#         fz = lambda alpha : f_alpha(alpha, x, d, theta, u1)
#         y = log_pdf(x, theta) - np.log(u1) # slice height
#         z_L = brentq(fz, a=-1e3, b=-1e-10)
#         z_R = brentq(fz, a=1e-10, b=1e3)

#         x_L = x + d*z_L
#         x_R = x + d*z_R
#         x_new = (1 - u2) * x_L + u2 * x_R

#         x = x_new
#         xs.append(x)
#         xLs.append(x_L)
#         xRs.append(x_R)
#         alphas.append(np.array([z_L,z_R]))

#     return np.array(xs), np.array(xLs), np.array(xRs), np.array(alphas)


# run forward pass
# xs, xLs, xRs, alphas = forwards2(S, params, x, us, ds_norm)

# x_range = np.arange(-5,5,0.01)
# g1 = 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * (x_range - _params[0])**2)
# g2 = 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-0.5 * (x_range - _params[1])**2)
# z1 = 0.5 
# plt.figure(figsize=[8,4])
# plt.plot(x_range, z1 * g1 + (1-z1) * g2)
# plt.hist(xs[:,0], 50, density=True)
