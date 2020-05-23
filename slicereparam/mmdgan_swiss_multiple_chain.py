from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np 
from jax import jit, grad
from jax import random
from jax import lax
from jax.ops import index, index_update
from jax.flatten_util import ravel_pytree
from functools import partial

from scipy.optimize import root_scalar, brentq
import matplotlib.pyplot as plt
from tqdm.auto import trange
from jax.scipy.special import expit as sigmoid

from data import load_mnist, save_images

def batch_normalize(activations):
    mbmean = np.mean(activations, axis=0, keepdims=True)
    return (activations - mbmean) / (np.std(activations, axis=0, keepdims=True) + 1)

def relu(x):       
    return np.maximum(0, x)

def rbf_kernel(x, y, sigma):
    """  
    x and y are both N samples by D data points
    """
    N, D = x.shape
    M, D = y.shape
    pairwise_diffs = (x[None,:,:] - y[:,None,:]).reshape((N*M,D))
    sqr_pairwise_diffs = np.sum(pairwise_diffs**2, axis=1)
    out = np.sum(np.exp( - 1.0 / 2.0 * np.outer(sqr_pairwise_diffs, 1.0 / sigma)),axis=1)
    return out # return sum? 

def visualize_2D(f, params=None, xmin=-5.0, xmax=5.0, dx=0.1, ax=None, vmin=0.0, vmax=0.2):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    import numpy as onp
    x1 = np.arange(xmin,xmax+dx,dx)
    x2 = np.arange(xmin,xmax+dx,dx)
    X1, X2 = np.meshgrid(x1, x2)
    Z = onp.zeros(X1.shape)
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            if params is None:
                Z[j,i] = f(np.array([x1[i], x2[j]]))
            else:
                Z[j,i] = f(np.array([x1[i], x2[j]]), params)
    # plt.imshow(Z / (np.sum(Z) * dx**2), extent=[xmin,xmax,xmin,xmax], origin="lower", vmin=0.0, vmax=0.1)
    # ax.imshow(np.exp(Z) / (np.sum(np.exp(Z)) * dx**2), extent=[xmin,xmax,xmin,xmax], origin="lower", vmin=vmin, vmax=vmax)
    ax.imshow(np.exp(Z), extent=[xmin,xmax,xmin,xmax], origin="lower")#, vmin=vmin, vmax=vmax)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([xmin, xmax])

def init_random_params(scale, layer_sizes, key):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    params = []
    for m, n in zip(layer_sizes[:-1], layer_sizes[1:]):
        key, *subkeys = random.split(key, 3)
        W = scale * random.normal(subkeys[0], (m, n))
        b = scale * random.normal(subkeys[1], (n, ))
        params.append([W,b])
    return params, key

# set up randomness
key = random.PRNGKey(1)

# Set up params
D = 2   # number of latent dimensions
D_out = 2 # dimensionality of data
H_energy = 256
scale = 0.001

energy_layer_sizes = [D, H_energy, H_energy, 1]
key, subkey = random.split(key)
_params, key = init_random_params(scale, energy_layer_sizes, subkey)
_params += [[0.0 * np.ones(D), np.log(2.0) * np.ones(D)]] # gaussian normalizer

def _log_pdf_batched(x, params):
    nn_params = params[:-1]
    mu, log_sigma_diag = params[-1]
    sigma_diag = np.exp(log_sigma_diag)
    inputs = x
    for W, b in nn_params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)   
        # inputs = np.tanh(outputs)
    outW, outb = nn_params[-1]
    out = np.dot(inputs, outW)+ outb
    return np.sum(out,axis=1) + np.sum(-0.5 * (x - mu) **2 / sigma_diag,axis=1)

def _log_pdf(x, params):
    nn_params = params[:-1]
    mu, log_sigma_diag = params[-1]
    sigma_diag = np.exp(log_sigma_diag)
    inputs = x
    for W, b in nn_params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)   
        # inputs = np.tanh(outputs)
    outW, outb = nn_params[-1]
    out = np.dot(inputs, outW)+ outb
    return np.sum(out) + np.sum(-0.5 * (x - mu) **2 / sigma_diag) 

params, unflatten = ravel_pytree(_params)
log_pdf = jit(lambda x, params : _log_pdf(x, unflatten(params)))
log_pdf_batched = jit(lambda x, params : _log_pdf_batched(x, unflatten(params)))

# compute necessary gradients
def log_pdf_theta(theta, x):    return log_pdf(x, theta)
def log_pdf_x(x, theta):        return log_pdf(x, theta)
def log_pdf_ad(x, theta, a, d): return log_pdf(x + a * d, theta)
grad_x = jit(grad(log_pdf_x))
grad_theta = jit(grad(log_pdf_theta))
grad_x_ad = jit(grad(log_pdf_ad))

def _total_loss(xs, ys, params, sigma=np.array([1.0,2.0,5.0,10.0,20.0,50.0])):
    k_xx = np.mean(rbf_kernel(xs, xs, sigma=sigma))
    k_xy = np.mean(rbf_kernel(xs, ys, sigma=sigma))
    k_yy = np.mean(rbf_kernel(ys, ys, sigma=sigma))
    return np.sqrt(k_xx - 2.0 * k_xy + k_yy)

loss = lambda x, y, params : _total_loss(x, y, unflatten(params))
total_loss = jit(lambda xs, ys, params : _total_loss(xs, ys, unflatten(params)))

# gradient of loss with respect to x
loss_grad_xs = jit(grad(total_loss))
loss_grad_params = jit(grad(lambda params, x, y : _total_loss(x, y, unflatten(params))))

@jit
def f_alpha(alpha, x, d, theta, u1):
    return log_pdf(x + alpha * d, theta) - log_pdf(x, theta) - np.log(u1)

@jit
def fa(x, alpha, d, theta, u1):
    return log_pdf(x + alpha * d, theta) - log_pdf(x, theta) - np.log(u1)

@jit
def fa_batched(x, alphas, d, theta, u1):
    return log_pdf_batched(x[None,:] + alphas[:,None] * d, theta) - log_pdf(x, theta) - np.log(u1)

@jit
def bisect_method(
    x, d, theta, u1,
    a, b, tol=1e-6, maxiter=100):

    i = 0
    init_val = [a, b, i]

    def cond_fun(val):
        a, b, i = val
        return np.sum((b-a) / 2.0) > tol

    def body_fun(val):

        a, b, i = val
        c = (a+b)/2.0
        # import ipdb; ipdb.set_trace()
        # temp_c = c+0.0
        sign_c = np.sign(fa(x, c, d, theta, u1))
        sign_a = np.sign(fa(x, a, d, theta, u1))
        sign_b = np.sign(fa(x, b, d, theta, u1))

        a = np.sum(c * np.maximum( sign_c * sign_a, 0.0) + \
            a * np.maximum( -1.0 * sign_c * sign_a, 0.0))
        b = np.sum(c * np.maximum( sign_c * sign_b, 0.0) + \
            b * np.maximum( -1.0 * sign_c * sign_b, 0.0))
        i = i+1

        val = [a, b, i]

        return val

    val = lax.while_loop(cond_fun, body_fun, init_val)
    a, b, i = val 
    c = (a+b)/2.0
    return c

@jit
def dual_fa(x, alpha1, alpha2, d, theta, u1):
    return [log_pdf(x + alpha1 * d, theta) - log_pdf(x, theta) - np.log(u1),
            log_pdf(x + alpha2 * d, theta) - log_pdf(x, theta) - np.log(u1)]


@jit
def dual_bisect_method(
    x, d, theta, u1,
    aL=-1e5, bL=-1e-5, aR=1e-5, bR=1e5,
    tol=1e-6, maxiter=100):

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

@jit
def choose_start(
    x, d, theta, u1,
    log_start = -2.0, log_space = 1.0 / 3.0):

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

# def forwards(S, theta, x, f_alpha, us, ds):
# def forwards(S, theta, x, us, ds):
#     xs = [x]
#     xLs = []
#     xRs = []
#     alphas = []

#     for s in range(S):

#         # import ipdb; ipdb.set_trace()
#         u1 = us[s,0]
#         u2 = us[s,1]
#         d = ds[s]

#         # fz = lambda alpha : f_alpha(alpha, x, d, theta, u1)
#         # z_L = brentq(fz, a=-1e3, b=-1e-10)
#         # z_R = brentq(fz, a=1e-10, b=1e3)

#         z_L = bisect_method(x, d, theta, u1, a=-25.0, b=-1e-10)
#         z_R = bisect_method(x, d, theta, u1, a=1e-10, b=25.0)
#         # z_L, z_R = dual_bisect_method(x, d, theta, u1, aL=-50.0, bL=-1e-10, aR=1e-10, bR=50.0)

#         x_L = x + d*z_L
#         x_R = x + d*z_R
#         x = (1 - u2) * x_L + u2 * x_R

#         xs.append(x)
#         xLs.append(x_L)
#         xRs.append(x_R)
#         alphas.append(np.array([z_L,z_R]))

#     return np.array(xs), np.array(xLs), np.array(xRs), np.array(alphas)

a_grid = np.concatenate((np.logspace(-2,1, 10), np.array([25.0])))
a_grid_total = np.concatenate((a_grid*-1.0, a_grid))

@jit
def fa_grid(x, d, theta, u1):
    fout = []
    for a in a_grid:
        fout.append(f_alpha(a, x, d, theta, u1))
    return np.array(fout)

@jit
def fa_batched(x, alphas, d, theta, u1):
    return log_pdf_batched(x[None,:] + alphas[:,None] * d, theta) - log_pdf(x, theta) - np.log(u1)


@jit
def fma_grid(x, d, theta, u1):
    fout = []
    for a in a_grid:
        fout.append(f_alpha(-1.0 * a, x, d, theta, u1))
    return np.array(fout)

# @jit
@jit
def forwards_step(x, theta, u1, u2, d):#, aL, bR):
    # z_L = bisect_method(x, d, theta, u1, a=-25.0, b=-1e-10)
    # z_R = bisect_method(x, d, theta, u1, a=1e-10, b=25.0)
    # z_L, z_R = dual_bisect_method(x, d, theta, u1, aL=-25.0, bL=-1e-10, aR=1e-10, bR=25.0)
    aL, bR = choose_start(x, d, theta, u1)
    z_L, z_R = dual_bisect_method(x, d, theta, u1, aL=aL, bL=-1e-10, aR=1e-10, bR=bR)
    x_L = x + d*z_L
    x_R = x + d*z_R
    x = (1 - u2) * x_L + u2 * x_R
    alphas = np.array([z_L, z_R])
    return x, x_L, x_R, alphas

len_a_grid = len(a_grid)
# @jit
def forwards(S, theta, x, us, ds):
    xs = [x]
    xLs = []
    xRs = []
    alphas = []
    for s in range(S):
        # aL=a_grid[np.where(fa(x, -a_grid, ds[s], theta, us[s,0])<0)[0][0]]*-1.0
        # aL=a_grid[np.where(fa(x, -a_grid, ds[s], theta, us[s,0])<0)[0][0]]*-1.0
        # aL=a_grid[np.where(fma_grid(x, ds[s], theta, us[s,0])<0)[0][0]]*-1.0
        # bR=a_grid[np.where(fa_grid(x, ds[s], theta, us[s,0])<0)[0][0]]
        # grid_vals = fa_batched(x, a_grid_total, ds[s], theta, us[s,0])
        # aL=a_grid[np.where(grid_vals[:len_a_grid]<0)[0][0]]*-1.0
        # bR=a_grid[np.where(grid_vals[len_a_grid:]<0)[0][0]]
        x, x_L, x_R, alpha = forwards_step(x, theta, us[s,0], us[s,1], ds[s])#, aL, bR)
        xs.append(x)
        xLs.append(x_L)
        xRs.append(x_R)
        alphas.append(alpha)
    return np.array(xs), np.array(xLs), np.array(xRs), np.array(alphas)


# from scipy.optimize import root 
# @jit
# def f_alphas_L(alphas, xs, ds, theta, u1s):
#     return f_alpha(alphas[:,None], xs, ds, theta, u1s) + np.maximum(alphas, 0)
# @jit
# def f_alphas_R(alphas, xs, ds, theta, u1s):
#     return f_alpha(alphas[:,None], xs, ds, theta, u1s) + np.minimum(alphas, 0)
# fzL = lambda alphas : f_alphas_L(alphas, xs[1:], norm_ds, theta, us[:,0])
# fzR = lambda alphas : f_alphas_R(alphas, xs[1:], norm_ds, theta, us[:,0])

# t1 = time.time()
# xL1 = root(fzL, x0=-1e-0*np.ones(S), method='df-sane').x
# xR1 = root(fzR, x0= 1e-0*np.ones(S), method='df-sane').x
# t2 = time.time()
# print((t2-t1)*1000)

# t1 = time.time()
# xL2 = []
# xR2 = []
# for s in range(S):
#     fz = lambda alpha : f_alpha(alpha, xs[s+1][None,:], norm_ds[s], theta, us[s,0])
#     xL2.append(bisect(fz, a=-1e3, b=-1e-10))
#     xR2.append(bisect(fz, a=1e-10, b=1e3))
# t2 = time.time()
# @jit

# bisect
# def bisect_method(
#     f, 
#     a, b, tol=2e-12, maxiter=100):

#     i = 0
#     init_val = [a, b, i]

#     def cond_fun(val):
#         a, b, i = val
#         return np.sum((a-b) / 2.0) < tol

#     def body_fun(val):

#         a, b, i = val
#         c = (a+b)/2.0
#         # import ipdb; ipdb.set_trace()
#         # temp_c = c+0.0
#         a = np.sum(c * np.maximum( np.sign(f(c)) * np.sign(f(a)), 0.0) + \
#             a * np.maximum( -1.0 * np.sign(f(c)) * np.sign(f(a)), 0.0))
#         b = np.sum(c * np.maximum( np.sign(f(c)) * np.sign(f(b)), 0.0) + \
#             b * np.maximum( -1.0 * np.sign(f(c)) * np.sign(f(b)), 0.0))
#         # if np.sign(f(c)) == np.sign(f(a)):
#         # if np.sign(f(c)) >0:
#             # a = c + 0
#         # else:
#             # b = c + 0
#         i = i+1

#         val = [a, b, i]

#         return val

#     val = lax.while_loop(cond_fun, body_fun, init_val)
#     a, b, i = val 
#     c = (a+b)/2.0
#     return c

# fzR = lambda alphas : f_alphas_R(alphas, xs, ds, theta, u1s)
# xL = root(fzL, x0=-1e-1*np.ones(N), method='broyden1').x

# function for backwards pass
def backwards(S, theta, us, ds, xs, xLs, xRs, alphas,
              grad_theta, grad_x, grad_x_ad, dL_dxs,
              loss_grad_params, ys, bS=0):

    D = xs[0].shape[0]
    dL_dtheta = np.zeros_like(theta)
    for s in range(S-1, -1+bS, -1):

        u1 = us[s,0]
        u2 = us[s,1]
        z_L = alphas[s][0]
        z_R = alphas[s][1]

        # compute loss for current sample
        dL_dx_s = dL_dxs[s] 

        # if not final sample, propagate loss from later samples
        if s < S-1:
            dL_dx_s = dL_dx_s + prev_dL_dx

        # compute gradients of xL and xR wrt theta
        L_grad_theta = -1.0 * (grad_theta(theta, xLs[s]) - grad_theta(theta, xs[s])) / np.dot(ds[s], grad_x_ad(xs[s], theta, z_L, ds[s]))
        R_grad_theta = -1.0 * (grad_theta(theta, xRs[s]) - grad_theta(theta, xs[s])) / np.dot(ds[s], grad_x_ad(xs[s], theta, z_R, ds[s]))

        # compute gradient dL / dtheta
        dLd = np.dot(dL_dx_s, ds[s]) # dot product between loss gradient and direction - this is used multiple times 
        dL_dtheta_s = u2 * dLd * R_grad_theta + (1-u2) * dLd * L_grad_theta
        dL_dtheta = dL_dtheta + dL_dtheta_s

        # propagate loss backwards : compute gradient times Jacobian of dx_s  / dx_{s-1}
        L_grad_x = -1.0 * ( grad_x_ad(xs[s], theta, z_L, ds[s]) - grad_x(xs[s], theta) ) / np.dot(ds[s], grad_x_ad(xs[s], theta, z_L, ds[s]))
        R_grad_x = -1.0 * ( grad_x_ad(xs[s], theta, z_R, ds[s]) - grad_x(xs[s], theta) ) / np.dot(ds[s], grad_x_ad(xs[s], theta, z_R, ds[s]))
        prev_dL_dx = dL_dx_s + u2 * dLd * R_grad_x + (1-u2) * dLd * L_grad_x

        # if you want to compute Jacobian dx_s / dx_{s-1}, you can use this line of code
        # J_xs = np.eye(D) + u2 * np.outer(ds[s], R_grad_x) + (1-u2) * np.outer(ds[s], L_grad_x)

    # TODO - do you want loss grad params for xs[1:] or xs?
    return dL_dtheta + loss_grad_params(theta, xs[1:], ys)

# @partial(jit, static_argnums={10,11,12})
# def backwards_step(theta, dL_dtheta, us, d, x, xL, xR, alphas, dL_dx, prev_dL_dx, grad_theta, grad_x, grad_x_ad):
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

    return dL_dtheta, prev_dL_dx

def backwards2(S, theta, us, ds, xs, xLs, xRs, alphas, dL_dxs, ys):
    dL_dtheta = np.zeros_like(theta)
    prev_dL_dx = np.zeros_like(xs[0])
    for s in range(S-1, -1, -1):
        dL_dtheta, prev_dL_dx = backwards_step(theta, dL_dtheta, us[s,:], ds[s], xs[s], 
                                               xLs[s], xRs[s], alphas[s], dL_dxs[s], prev_dL_dx)
    return dL_dtheta + loss_grad_params(theta, xs[1:], ys)

# @partial(jit, static_argnums={10,11,12})
# def backwards3(S, theta, us, ds, xs, xLs, xRs, alphas, dL_dxs, ys, grad_theta, grad_x, grad_x_ad, loss_grad_params):
@jit
def backwards3(S, theta, us, ds, xs, xLs, xRs, alphas, dL_dxs, ys):

    dL_dtheta = np.zeros_like(theta)
    prev_dL_dx = np.zeros_like(xs[0])
    init_val = [S-1, dL_dtheta, prev_dL_dx]

    def cond_fun(val):
        return val[0] > -1

    def body_fun(val):
        s = val[0]
        dL_dtheta, prev_dL_dx = val[1:] 
        dL_dtheta, prev_dL_dx = backwards_step(theta, dL_dtheta, us[s,:], ds[s], xs[s], 
                                               xLs[s], xRs[s], alphas[s], dL_dxs[s], prev_dL_dx)
        val[0] -= 1
        return [val[0], dL_dtheta, prev_dL_dx]

    val = lax.while_loop(cond_fun, body_fun, init_val)
    dL_dtheta = val[1]
    return dL_dtheta

# test functions
S = 64 # number of samples
key, *subkeys = random.split(key, 4)
us = random.uniform(subkeys[0], (S,2))
ds = random.normal(subkeys[1], (S,D))
ds_norm = np.array([d / np.linalg.norm(d) for d in ds])
x = 0.1 * random.normal(subkeys[2], (D,)) # initial x 

# run forward pass
# xs, xLs, xRs, alphas = forwards(S, params, x, f_alpha, us, ds_norm)
xs, xLs, xRs, alphas = forwards(S, params, x, us, ds_norm)

# run backward pass
key, *subkeys = random.split(key)
ys = random.normal(key, (S, D_out))
dL_dxs = loss_grad_xs(xs[1:], ys, params)
# import time
# t1 = time.time()
dL_dtheta0 = backwards(S, params, us, ds_norm, xs, xLs, xRs, alphas, grad_theta, grad_x, grad_x_ad, dL_dxs, loss_grad_params, ys)
# t2 = time.time()
# print((t2-t1))
# t1 = time.time()
# dL_dtheta = backwards2(S, params, us, ds_norm, xs, xLs, xRs, alphas, dL_dxs, ys)
# t2 = time.time()
# print((t2-t1))
# t1 = time.time()
dL_dtheta2 = backwards3(S, params, us, ds_norm, xs, xLs, xRs, alphas, dL_dxs, ys)
# t2 = time.time()
# print((t2-t1))
# t1 = time.time(); dL_dtheta2 = backwards3(S, params, us, ds_norm, xs, xLs, xRs, alphas, dL_dxs, ys); t2 = time.time(); print(t2-t1)
print("Implicit: ", dL_dtheta2)
print("Norm b/w backwards and backwards3: ", np.linalg.norm(dL_dtheta0-dL_dtheta2))

# compute gradient via finite differences
# dx = 1e-5
# theta = params+0
# M = theta.shape[0]
# dthetas = np.zeros_like(theta)
# print(M)
# for m, v in enumerate(np.eye(M)):
#     print(m)
#     theta1 = theta - dx * v
#     theta2 = theta + dx * v
#     xs1, xLs1, xRs1, alphas1 = forwards(S, theta1, x, f_alpha, us, ds_norm)
#     xs2, xLs2, xRs2, alphas2 = forwards(S, theta2, x, f_alpha, us, ds_norm)
#     loss1 = total_loss(np.array(xs1[1:]), ys, theta1)
#     loss2 = total_loss(np.array(xs2[1:]), ys, theta2)
#     dthetas = dthetas + (loss2 - loss1) / (2.0 * dx) * v

# print("Numerical: ", dthetas)
# print("MSE: ", np.mean((dL_dtheta - dthetas)**2)) 

# load data
from sklearn.datasets import make_swiss_roll
N = 100000
noise = 0.25
X, _ = make_swiss_roll(N, noise)
X /= 7.5
X = X[:,[0,2]]

# # optimize parameters!
theta = params+0.0
# d = np.load("mmdgan_swiss.npz")
# theta = d["theta"]
M = theta.shape[0]
thetas = [theta]
xs = [x]
losses = [0.0]

# set up iters
S = 250
num_iters = int(np.ceil(N / S))
num_chains = 25
Sc = int(S / num_chains)

@jit
def batch_indices(iter):
    idx = iter % num_iters
    return slice(idx * S, (idx+1) * S) # S is batch size

# set up randomness
@jit
def generate_randomness(key):
    key, *subkeys = random.split(key, 5)
    us = random.uniform(subkeys[0], (S,2))
    ds = random.normal(subkeys[1], (S,D))
    ds_norm = ds / np.sqrt(np.sum(ds**2, axis=1))[:,None]
    data_idx = random.randint(subkeys[2], (S, ), 0, N)
    x0 = random.normal(subkeys[3], (num_chains, D))
    return us, ds_norm, data_idx, x0, key

burn_in = 3
@jit
def generate_randomness_burnin(key):
    key, *subkeys = random.split(key, 4)
    us = random.uniform(subkeys[0], (burn_in*num_chains,2))
    ds = random.normal(subkeys[1], (burn_in*num_chains,D))
    ds_norm = ds / np.sqrt(np.sum(ds**2, axis=1))[:,None]
    x0 = random.normal(subkeys[2], (num_chains, D))
    return us, ds_norm, x0, key

# for ADAM
adam_iter = 0.0
m = np.zeros(len(theta))
v = np.zeros(len(theta))
b1 = 0.5
b2 = 0.9
step_size = 0.001
eps=10**-8

@jit
def adam_step(theta, dL_dtheta, m, v, adam_iter):
    m = (1 - b1) * dL_dtheta      + b1 * m  # First  moment estimate.
    v = (1 - b2) * (dL_dtheta**2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1**(adam_iter + 1))            # Bias correction.
    vhat = v / (1 - b2**(adam_iter + 1))
    theta = theta - step_size*mhat/(np.sqrt(vhat) + eps)
    adam_iter = adam_iter + 1
    return theta, m, v, adam_iter

def plot_update(xs, theta, key):
    key, subkey = random.split(key)
    rand_idx = random.randint(subkey, (8, ), 0, S)
    plt.clf()
    plt.subplot(331)
    plt.plot(xs[:,0], xs[:,1])
    plt.subplot(332)
    plt.imshow(x_images[0].reshape((28,28)), cmap="gray", vmin=0.0, vmax=1.0)    
    plt.subplot(333)
    plt.imshow(x_images[1].reshape((28,28)), cmap="gray", vmin=0.0, vmax=1.0)    
    plt.subplot(334)
    plt.imshow(x_images[2].reshape((28,28)), cmap="gray", vmin=0.0, vmax=1.0)    
    plt.subplot(335)
    plt.imshow(x_images[3].reshape((28,28)), cmap="gray", vmin=0.0, vmax=1.0)    
    plt.subplot(336)
    plt.imshow(x_images[4].reshape((28,28)), cmap="gray", vmin=0.0, vmax=1.0)    
    plt.subplot(337)
    plt.imshow(x_images[5].reshape((28,28)), cmap="gray", vmin=0.0, vmax=1.0)    
    plt.subplot(338)
    plt.imshow(x_images[6].reshape((28,28)), cmap="gray", vmin=0.0, vmax=1.0)    
    plt.subplot(339)
    plt.imshow(x_images[7].reshape((28,28)), cmap="gray", vmin=0.0, vmax=1.0)   
    plt.pause(0.01)
    return key 

fig = plt.figure(figsize=[8,6])
num_epochs = 3
pbar = trange(num_iters*num_epochs)
pbar.set_description("Loss: {:.1f}".format(losses[0]))
import time 
for epoch in range(num_epochs):
    for i in range(num_iters):

        us, norm_ds, x0b, key = generate_randomness_burnin(key)
        # burn in 
        x, _, _, _ = forwards(burn_in, theta, x0b[0], us[:burn_in], norm_ds[:burn_in])
        x0 = x[-1]
        for j in range(1, num_chains):
            x, _, _, _ = forwards(burn_in, theta, x0b[j], us[j*burn_in:(j+1)*burn_in], norm_ds[j*burn_in:(j+1)*burn_in])
            x0 = np.vstack((x0, x[-1]))

        us, norm_ds, data_idx, x0, key = generate_randomness(key)

        ys = X[data_idx]

        # forwards
        x, xLs, xRs, alphas = forwards(Sc, theta, x0[0], us[:Sc], norm_ds[:Sc])
        xs = x[1:]
        for j in range(1,num_chains):
            x, xL, xR, alpha = forwards(Sc, theta, x0[j], us[j*Sc:(j+1)*Sc], norm_ds[j*Sc:(j+1)*Sc])
            xs = np.vstack((xs, x[1:]))
            xLs = np.vstack((xLs, xL))
            xRs = np.vstack((xRs, xR))
            alphas = np.vstack((alphas, alpha))

        # backwards pass
        dL_dxs = loss_grad_xs(xs, ys, theta)
        dL_dtheta = loss_grad_params(theta, xs, ys)

        # backwards
        for j in range(num_chains):
            xs_j = np.vstack((x0[j], xs[j*Sc:(j+1)*Sc]))
            dL_dtheta = dL_dtheta + backwards3(Sc, theta, us[j*Sc:(j+1)*Sc], norm_ds[j*Sc:(j+1)*Sc], 
                                                xs_j, xLs[j*Sc:(j+1)*Sc], xRs[j*Sc:(j+1)*Sc], 
                                                alphas[j*Sc:(j+1)*Sc], dL_dxs[j*Sc:(j+1)*Sc], ys)


        # import ipdb; ipdb.set_trace()
        # ADAM
        theta, m, v, adam_iter = adam_step(theta, dL_dtheta, m, v, adam_iter)

        if np.mod(i, 100) == 0:
            plt.clf()
            plt.plot(ys[:,0], ys[:,1], 'r.', markersize=5, label="True")
            plt.plot(xs[:,0], xs[:,1], 'b.', markersize=5, label="Gen")
            plt.legend()
            plt.pause(0.01)

            # key = plot_update(xs, theta, key)
            # thetas.append(theta)

        losses.append(total_loss(xs[1:], ys, theta))


        pbar.set_description("Loss: {:.1f}".format(losses[-1]))
        pbar.update()



pbar.close()

# thetas_plot = np.array(thetas)

# plt.savefig("optimize_moments_dim" + str(D) + "_samples" + str(S) + ".png")

# gauss_log_pdf = lambda x : -0.5 * (x - xstar).T @ np.linalg.inv(Cov) @ (x - xstar)
np.savez("mmdgan_swiss_multiple_chain_weights.npz", theta=theta, losses=np.array(losses), m=m, v=v, adam_iter=adam_iter)

# xmin = -3.0
# xmax =3.0
# plt.figure()
# visualize_2D(log_pdf, params, xmin=xmin, xmax=xmax, vmin=0.0, vmax=0.17, dx=0.1)
# plt.title("Init")
# plt.plot(X[:1000,0], X[:1000,1], 'r.', markersize=2.5)


# xmin = -3.0
# xmax =3.0
# plt.figure()
# visualize_2D(log_pdf, theta, xmin=xmin, xmax=xmax, vmin=0.0, vmax=0.17, dx=0.1)
# plt.title("Generative, Iteration: " + str(len(losses)))
# plt.plot(X[:1000,0], X[:1000,1], 'r.', markersize=2.5)

xmin = -3.0
xmax =3.0
plt.figure()
ax1=plt.subplot(121)
visualize_2D(log_pdf, params, xmin=xmin, xmax=xmax, vmin=0.0, vmax=0.15, dx=0.1,ax=ax1)
plt.plot(X[:2500,0], X[:2500,1], 'r.', markersize=2.5)
plt.title("Init")
ax2=plt.subplot(122)
visualize_2D(log_pdf, theta, xmin=xmin, xmax=xmax, vmin=0.0, vmax=0.15, dx=0.1, ax =ax2)
plt.plot(X[:2500,0], X[:2500,1], 'r.', markersize=2.5)
plt.title("Generative, Iteration: " + str(len(losses)))

xmin = -3.0
xmax =3.0
plt.figure()
visualize_2D(log_pdf, theta, xmin=xmin, xmax=xmax, vmin=0.0, vmax=0.15, dx=0.1)
plt.title("Generative, Iteration: " + str(len(losses)))
plt.plot(X[:500,0], X[:500,1], 'r.', markersize=2.5)


plt.figure()
plt.subplot(221)
plt.plot(xs[:,0], xs[:,1])
# test sample a lot of xs
# S2 = 1000
# us = npr.rand(S2, 2)
# ds = npr.randn(S2, D)
# norm_ds = np.array([d / np.linalg.norm(d) for d in ds])
# x0 = xs[-1]
# xs2, xLs, xRs, alphas = forwards(S2, theta, x0, f_alpha, us, norm_ds)
idx=-1
images = _generate(xs_new, unflatten(theta))
idx+=1
plt.figure()
plt.imshow(images[idx].reshape((28,28)), cmap="gray", vmin=0.0, vmax=1.0)

key, subkey = random.split(key)
# jit_generate=jit(_generate)
images = _generate(random.normal(subkey, (1, D)), unflatten(theta))
# images = jit_generate(random.normal(subkey, (1, D)), unflatten(theta))
plt.figure()
plt.imshow(images[0].reshape((28,28)), cmap="gray", vmin=0.0, vmax=1.0)