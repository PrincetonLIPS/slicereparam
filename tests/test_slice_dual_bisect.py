from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np 
from jax import jit, grad, vmap
from jax import random
from jax import lax
from jax.ops import index, index_update
from jax.flatten_util import ravel_pytree
from functools import partial

from scipy.optimize import root_scalar, brentq
import matplotlib.pyplot as plt
from tqdm.auto import trange
from jax.scipy.special import expit as sigmoid
from jax.scipy.special import logsumexp

import time

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
    tol=1e-6, maxiter=100):

    i = maxiter-1.0
    bracket_vals = [aL, bL, aR, bR]
    sign_aL = np.sign(fa(x, aL, d, theta, u1))
    sign_bL = np.sign(fa(x, bL, d, theta, u1))
    sign_aR = np.sign(fa(x, aR, d, theta, u1))
    sign_bR = np.sign(fa(x, bR, d, theta, u1))

    bracket_sign_fvals = [sign_aL, sign_bL, sign_aR, sign_bR]
    init_val = [bracket_vals, bracket_sign_fvals, i]

    def cond_fun(val):
        bracket_vals, bracket_sign_fvals, i = val
        aL, bL, aR, bR = bracket_vals 
        return np.sum(np.abs(bL-aL)) + np.sum(np.abs(bR-aR)) + 100 * np.minimum(i, 0.0) > tol

    def body_fun(val):

        # unpack val
        bracket_vals, bracket_sign_fvals, i = val
        aL, bL, aR, bR = bracket_vals 
        sign_aL, sign_bL, sign_aR, sign_bR = bracket_sign_fvals

        # new center points
        cL = (aL+bL)/2.0
        cR = (aR+bR)/2.0

        # L
        sign_cL = np.sign(fa(x, cL, d, theta, u1))
        aL = lax.cond(sign_cL * sign_aL > 0, (), lambda _ : cL, (), lambda _ : aL)
        bL = lax.cond(sign_cL * sign_bL > 0, (), lambda _ : cL, (), lambda _ : bL)
        sign_aL = lax.cond(sign_cL * sign_aL > 0, (), lambda _ : sign_cL, (), lambda _ : sign_aL)
        sign_bL = lax.cond(sign_cL * sign_bL > 0, (), lambda _ : sign_cL, (), lambda _ : sign_bL)

        # R
        sign_cR = np.sign(fa(x, cR, d, theta, u1))
        aR = lax.cond(sign_cR * sign_aR > 0, (), lambda _ : cR, (), lambda _ : aR)
        bR = lax.cond(sign_cR * sign_bR > 0, (), lambda _ : cR, (), lambda _ : bR)
        sign_aR = lax.cond(sign_cR * sign_aR > 0, (), lambda _ : sign_cR, (), lambda _ : sign_aR)
        sign_bR = lax.cond(sign_cR * sign_bR > 0, (), lambda _ : sign_cR, (), lambda _ : sign_bR)

        i = i - 1
        bracket_vals = [aL, bL, aR, bR]
        bracket_sign_fvals = [sign_aL, sign_bL, sign_aR, sign_bR]
        val = [bracket_vals, bracket_sign_fvals, i]

        return val

    val = lax.while_loop(cond_fun, body_fun, init_val)

    # unpack val
    bracket_vals, bracket_sign_fvals, i = val
    aL, bL, aR, bR = bracket_vals 

    # new center points
    cL = (aL+bL)/2.0
    cR = (aR+bR)/2.0

    return [cL, cR]

# a_grid = np.concatenate((np.logspace(-3,1, 25), np.array([25.0])))
@jit
def choose_start(
    x, d, theta, u1,
    log_start = -3.0, log_space = 1.0 / 6.0):

    i = 0
    aL = -1.0 * np.power(10.0, log_start + i * log_space)
    bR = np.power(10.0, log_start + i * log_space)
    aL_val = fa(x, aL, d, theta, u1)
    bR_val = fa(x, bR, d, theta, u1)
    init_val = [aL, bR, aL_val, bR_val, i]

    def cond_fun(val):
        aL, bR, aL_val, bR_val, i = val
        return np.maximum(aL_val, 0.0) + np.maximum(bR_val, 0.0) > 0.0

    def body_fun(val):

        aL, bR, aL_val, bR_val, i = val
        i = i+1
        sign_aL = np.sign(aL_val)
        sign_bR = np.sign(bR_val)
        aL = np.sum(-1.0 * np.power(10.0, log_start + i * log_space) * np.maximum(sign_aL, 0.0) \
                + aL * np.maximum(-1.0 * sign_aL, 0.0))
        bR = np.sum(np.power(10.0, log_start + i * log_space) * np.maximum(sign_bR, 0.0) \
                + bR * np.maximum(-1.0 * sign_bR, 0.0))
        aL_val = fa(x, aL, d, theta, u1)
        bR_val = fa(x, bR, d, theta, u1)
        val = [aL, bR, aL_val, bR_val, i]

        return val

    val = lax.while_loop(cond_fun, body_fun, init_val)
    aL, bR, aL_val, bR_val, i = val
    return [aL, bR]

@jit
def forwards_step(x, theta, u1, u2, d):#, aL, bR):
    aL, bR = choose_start(x, d, theta, u1)
    z_L, z_R = dual_bisect_method(x, d, theta, u1, aL=aL, bL=-1e-10, aR=1e-10, bR=bR)
    x_L = x + d*z_L
    x_R = x + d*z_R
    x = (1 - u2) * x_L + u2 * x_R
    alphas = np.array([z_L, z_R])
    return x, x_L, x_R, alphas

vmapped_forwards_step = jit(vmap(forwards_step, (0,None,0,0,0)))

# @jit
def forwards(S, theta, x, us, ds):
    xs = [x]
    xLs = []
    xRs = []
    alphas = []
    for s in range(S):
        # x, x_L, x_R, alpha = forwards_step(x, theta, us[s,:,0], us[s,:,1], ds[s])#, aL, bR)
        x, x_L, x_R, alpha = vmapped_forwards_step(x, theta, us[s,:,0], us[s,:,1], ds[s])#, aL, bR)
        xs.append(x)
        xLs.append(x_L)
        xRs.append(x_R)
        alphas.append(alpha)
    return np.array(xs), np.array(xLs), np.array(xRs), np.array(alphas)

var1 = 1.0
var2 = 2.0
def _log_pdf(x, params):

    mu1 = params[0]
    mu2 = params[1]

    log1 = -0.5 * (x - mu1)**2 / var1 - 0.5 * np.sqrt(2.0 * np.pi * var1)
    log2 = -0.5 * (x - mu2)**2 / var2 - 0.5 * np.sqrt(2.0 * np.pi * var2)
    return np.sum(logsumexp(np.array([log1,log2]),axis=0))

# sample from exponential (positive support)
def _log_pdf(x, params):
    return np.sum(-np.log(params[0]) - 1.0 * np.maximum(x, 0.0) * params[0] - np.sign(np.minimum(x, 0)) * np.log(1e-25))
# _params = [1.0]

# compute necessary gradients
def log_pdf_theta(theta, x):    return log_pdf(x, theta)
def log_pdf_x(x, theta):        return log_pdf(x, theta)
def log_pdf_ad(x, theta, a, d): return log_pdf(x + a * d, theta)
grad_x = jit(grad(log_pdf_x))
grad_theta = jit(grad(log_pdf_theta))
grad_x_ad = jit(grad(log_pdf_ad))

_params = [-3.0, 3.0]
params, unflatten = ravel_pytree(_params)
log_pdf = jit(lambda x, params : _log_pdf(x, unflatten(params)))

D = 1
key = random.PRNGKey(5)

S = 200 # number of samples
num_chains = 200 # number of chains
key, *subkeys = random.split(key, 4)
us = random.uniform(subkeys[0], (S,num_chains,2))
ds = random.normal(subkeys[1], (S*num_chains,D))
ds_norm = ds / np.sqrt(np.sum(ds**2, axis=1))[:,None]
ds_norm = ds_norm.reshape((S, num_chains, D))
x = 1.0 * random.normal(subkeys[2], (num_chains,D)) # initial x 
# x = np.abs(1.0 * random.normal(subkeys[2], (num_chains,D))) + 0.5 # initial x for exponential (pos support)

# run forward pass
t1 = time.time()
xs, xLs, xRs, alphas = forwards(S, params, x, us, ds_norm)
t2 = time.time()
print(t2-t1)
xs_plot = np.reshape(xs[1:,:,0], S*num_chains)
dx = 0.01
x_range = np.arange(-10,10,dx)
# g1 = 1.0 / np.sqrt(2.0 * np.pi * var1) * np.exp(-0.5 * (x_range - _params[0])**2 / var1)
# g2 = 1.0 / np.sqrt(2.0 * np.pi * var2) * np.exp(-0.5 * (x_range - _params[1])**2 / var2)
# z1 = 0.5 

pdf = np.array([np.exp(log_pdf(x, params)) for x in x_range])
normalizer = np.sum(pdf)*dx

plt.figure(figsize=[8,4])
plt.subplot(211)
plt.plot(x_range, pdf / normalizer)
plt.hist(xs_plot, 80, density=True);
plt.title("with step out bracket")


@jit
def forwards_step_without_choose_start(x, theta, u1, u2, d):#, aL, bR):
    z_L, z_R = dual_bisect_method(x, d, theta, u1, bL=-1e-10, aR=1e-10)
    x_L = x + d*z_L
    x_R = x + d*z_R
    x = (1 - u2) * x_L + u2 * x_R
    alphas = np.array([z_L, z_R])
    return x, x_L, x_R, alphas

vmapped_forwards_step_without_choose_start = jit(vmap(forwards_step_without_choose_start, (0,None,0,0,0)))

# @jit
def forwards_without_choose_start(S, theta, x, us, ds):
    xs = [x]
    xLs = []
    xRs = []
    alphas = []
    for s in range(S):
        x, x_L, x_R, alpha = vmapped_forwards_step_without_choose_start(x, theta, us[s,:,0], us[s,:,1], ds[s])#, aL, bR)
        xs.append(x)
        xLs.append(x_L)
        xRs.append(x_R)
        alphas.append(alpha)
    return np.array(xs), np.array(xLs), np.array(xRs), np.array(alphas)

# run forward pass
xs2, xLs, xRs, alphas = forwards_without_choose_start(S, params, x, us, ds_norm)
xs_plot2 = np.reshape(xs2[1:,:,0], S*num_chains)
# plt.figure(figsize=[8,4])
plt.subplot(212)
plt.plot(x_range, pdf / normalizer)
plt.hist(xs_plot2, 80, density=True, label="w/o choose start");
plt.title("without step out bracket")

plt.figure()
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.plot(x_range, pdf / normalizer)
    plt.hist(xs[1:,i,0], 20, density=True)
    plt.xticks([])
    plt.yticks([])


## backwrads sample
@jit
def backwards_step(theta, dL_dtheta, us, d, x, xL, xR, alphas, dL_dx, prev_dL_dx):

    u1 = us[0]
    u2 = us[1]
    z_L = alphas[0]
    z_R = alphas[1]

    # compute loss for current sample
    # set prev_dL_dx to zero at first
    dL_dx_s = dL_dx + prev_dL_dx

    # compute gradients of xL and xR wrt theta
    L_grad_theta = -1.0 * (grad_theta(theta, xL) - grad_theta(theta, x)) / np.dot(d, grad_x_ad(x, theta, z_L, d))
    R_grad_theta = -1.0 * (grad_theta(theta, xR) - grad_theta(theta, x)) / np.dot(d, grad_x_ad(x, theta, z_R, d))

    # compute gradient dL / dtheta
    dLd = np.dot(dL_dx_s, d) # dot product between loss gradient and direction - this is used multiple times 
    dL_dtheta_s = u2 * dLd * R_grad_theta + (1-u2) * dLd * L_grad_theta
    dL_dtheta = dL_dtheta + dL_dtheta_s

    # propagate loss backwards : compute gradient times Jacobian of dx_s  / dx_{s-1}
    L_grad_x = -1.0 * ( grad_x_ad(x, theta, z_L, d) - grad_x(x, theta) ) / np.dot(d, grad_x_ad(x, theta, z_L, d))
    R_grad_x = -1.0 * ( grad_x_ad(x, theta, z_R, d) - grad_x(x, theta) ) / np.dot(d, grad_x_ad(x, theta, z_R, d))
    prev_dL_dx = dL_dx_s + u2 * dLd * R_grad_x + (1-u2) * dLd * L_grad_x

    J_x = np.eye(D) + u2 * np.outer(d, R_grad_x) + (1-u2) * np.outer(d, L_grad_x)

    return dL_dtheta, prev_dL_dx, J_x

def get_jacobians(S, theta, us, ds, xs, xLs, xRs, alphas, dL_dxs):
    dL_dtheta = np.zeros_like(theta)
    prev_dL_dx = np.zeros_like(xs[0])
    J_xs = []
    for s in range(S-1, -1, -1):
        dL_dtheta, prev_dL_dx, J_x = backwards_step(theta, dL_dtheta, us[s,:], ds[s], xs[s], 
                                               xLs[s], xRs[s], alphas[s], dL_dxs[s], prev_dL_dx)
        J_xs.append(J_x)
    return dL_dtheta, J_xs


S = 100 # number of samples
num_chains = 1 # number of chains
key, *subkeys = random.split(key, 4)
us = random.uniform(subkeys[0], (S,num_chains,2))
ds = random.normal(subkeys[1], (S*num_chains,D))
ds_norm = ds / np.sqrt(np.sum(ds**2, axis=1))[:,None]
ds_norm = ds_norm.reshape((S, num_chains, D))
x = 0.0 + 3.0 * random.normal(subkeys[2], (num_chains,D)) # initial x 
xs, xLs, xRs, alphas = forwards(S, params, x, us, ds_norm)

chain_number = 0
dL_dtheta, J_xs = get_jacobians(S, params, us[:,chain_number,:], ds_norm[:,chain_number,:],
                                xs[:,chain_number,:], xLs[:,chain_number,:], xRs[:,chain_number,:],
                                alphas[:,chain_number,:], np.zeros((S,D)))
J_xs.reverse()
J_xs = np.array(J_xs)
S_plot = 100
plt.figure()
plt.subplot(211)
plt.plot(np.arange(S_plot+1), xs[:S_plot+1,chain_number,0])
plt.ylabel("$x_n$")
# plt.subplot(312)
# plt.plot(np.arange(S_plot)+1, np.abs((J_xs[:S_plot,0])))
# plt.xlabel("iteration")
plt.subplot(212)
plt.plot(np.arange(S_plot)+1, np.abs(np.cumprod(J_xs[:S_plot,0])))
plt.xlabel("iteration n")
plt.ylabel("$| dx_n / dx_0 |$")