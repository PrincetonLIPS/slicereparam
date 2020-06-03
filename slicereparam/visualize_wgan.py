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

from data import load_mnist, save_images

def batch_normalize(activations):
    mbmean = np.mean(activations, axis=0, keepdims=True)
    return (activations - mbmean) / (np.std(activations, axis=0, keepdims=True) + 1)

def relu(x):       
    return np.maximum(0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def swish(x):
    return np.multiply(x, sigmoid(x))

dataset = "swiss"
# dataset = "circle"
# dataset = "mog"

def visualize_2D(f, params=None, xmin=-5.0, xmax=5.0, dx=0.1, ax=None, vmin=0.0, vmax=0.2, log=False):
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
    if log:
        ax.imshow(Z, extent=[xmin,xmax,xmin,xmax], origin="lower")
    else:
        ax.imshow(np.exp(Z) / (np.sum(np.exp(Z)) * dx**2), extent=[xmin,xmax,xmin,xmax], origin="lower", vmin=vmin, vmax=vmax)
        # ax.imshow(np.exp(Z), extent=[xmin,xmax,xmin,xmax], origin="lower")#, vmin=vmin, vmax=vmax)
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
key = random.PRNGKey(1313)

# Set up params
D = 2   # number of latent dimensions
D_out = 2 # dimensionality of data
H_energy = 256
scale = 0.001

energy_layer_sizes = [D, H_energy, H_energy, H_energy, 1]
key, subkey = random.split(key)
_params, key = init_random_params(scale, energy_layer_sizes, subkey)
_params += [[0.0 * np.ones(D), np.log(0.01) * np.ones(D)]] # gaussian normalizer

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

H_disc = 512
disc_layer_sizes = [D, H_disc, H_disc, H_disc, 1]
key, subkey = random.split(key)
_disc_params, key = init_random_params(scale, disc_layer_sizes, subkey)

def _discriminator(x, params):
    inputs = x
    for W, b in params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)   
        # inputs = np.tanh(outputs)
    outW, outb = params[-1]
    out = np.dot(inputs, outW)+ outb
    return out


disc_params, unflatten_disc = ravel_pytree(_disc_params)
discriminator = jit(lambda x, params : _discriminator(x, unflatten_disc(params)))

# compute necessary gradients
def log_pdf_theta(theta, x):    return log_pdf(x, theta)
def log_pdf_x(x, theta):        return log_pdf(x, theta)
def log_pdf_ad(x, theta, a, d): return log_pdf(x + a * d, theta)
grad_x = jit(grad(log_pdf_x))
grad_theta = jit(grad(log_pdf_theta))
grad_x_ad = jit(grad(log_pdf_ad))

grad_disc_x = jit(grad(lambda x, params : np.sum(_discriminator(x, unflatten_disc(params)))))
def discriminator_phi(params, x): return np.sum(discriminator(x, params))
grad_disc_phi = jit(grad(discriminator_phi))

def L_func(params, xtilde, x, xhat, lambda_val):
    L = np.mean(discriminator(xtilde, params) - discriminator(x, params))
    grad_penalty = lambda_val * np.mean( (np.sqrt(np.sum(grad_disc_x(xhats, params)**2, axis=1)+1e-12) - 1.0)**2 )
    return L + grad_penalty
grad_L_func = jit(grad(L_func))

@jit
def total_loss(xs, disc_params):
    return - 1.0 * np.mean(discriminator(xs, disc_params))

# gradient of loss with respect to x
loss_grad_xs = jit(grad(total_loss))
loss_grad_params = jit(grad(lambda params, x, disc_params : total_loss(x, disc_params)))

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

        i = i - 1
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

a_grid = np.concatenate((np.logspace(-2,1, 10), np.array([25.0])))
a_grid_total = np.concatenate((a_grid*-1.0, a_grid))

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
# S steps
# us is S x N chains x D
# ds is S x N chains x D
# x is N chains x D
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

vmapped_backwards = jit(vmap(backwards3, (None, None, 0, 0, 0, 0, 0, 0, 0, None)))

# load data
if dataset is "swiss":
    from sklearn.datasets import make_swiss_roll
    N = 100000
    noise = 0.25
    X, _ = make_swiss_roll(N, noise)
    X /= 7.5
    X = X[:,[0,2]]

elif dataset is "circle":
    N = 100000
    noise = 0.01
    rs = np.array([0.5,1.0,1.5,2.0])
    key, subkey = random.split(key)
    # zs = random.randint(subkey, (N, ), minval=0, maxval=4)
    # r_zs = rs[zs]
    r_zs = np.ones((N,))
    key, subkey = random.split(key)
    angles = 2.0 * np.pi * random.uniform(subkey, (N,))
    X = r_zs[:,None] * np.hstack((np.cos(angles)[:,None], np.sin(angles[:,None])))
    key, subkey = random.split(key)
    X = X + np.sqrt(noise) * random.normal(subkey, (N, D))

elif dataset is "mog":

    mus = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    N = 100000
    num_classes = mus.shape[0]**2
    N_class = int(N / num_classes)
    X = np.empty([0,D])
    noise = 0.01
    for mu1 in mus:
        for mu2 in mus:
            key, subkey = random.split(key)
            X_new = np.array([mu1, mu2]) + np.sqrt(noise) * random.normal(subkey, (N_class, D))
            X = np.vstack((X, X_new))
    X = random.shuffle(key, X, axis=0)
    X = X / 2.828

# # optimize parameters!
theta = params+0.0
phi = disc_params + 0.0
M = theta.shape[0]
losses = [0.0]

# set up iters
S = 250
num_iters = int(np.ceil(N / S))
num_chains = 250
Sc = int(S / num_chains)

@jit
def batch_indices(iter):
    idx = iter % num_iters
    return slice(idx * S, (idx+1) * S) # S is batch size

# set up randomness
@jit
def generate_randomness(key):
    key, *subkeys = random.split(key, 4)
    us = random.uniform(subkeys[0], (Sc,num_chains,2))
    ds = random.normal(subkeys[1], (Sc*num_chains,D))
    ds_norm = ds / np.sqrt(np.sum(ds**2, axis=1))[:,None]
    ds_norm = ds_norm.reshape((Sc, num_chains, D))
    x0 = random.normal(subkeys[2], (num_chains, D))
    return us, ds_norm, x0, key

burn_in = 4

@jit
def generate_randomness_burnin(key):
    key, *subkeys = random.split(key, 4)
    us = random.uniform(subkeys[0], (burn_in+Sc,num_chains,2))
    ds = random.normal(subkeys[1], ( (burn_in+Sc) * num_chains,D))
    ds_norm = ds / np.sqrt(np.sum(ds**2, axis=1))[:,None]
    ds_norm = ds_norm.reshape((burn_in+Sc, num_chains, D))
    x0 = random.normal(subkeys[2], (num_chains, D))
    return us, ds_norm, x0, key

burn_in_critic = 4
@partial(jit, static_argnums={0})
def generate_randomness_var(num_chains, key):
    S_total = (Sc+burn_in_critic) * num_chains
    key, *subkeys = random.split(key, 4)
    us = random.uniform(subkeys[0], (Sc+burn_in_critic,num_chains,2))
    ds = random.normal(subkeys[1], (S_total,D))
    ds_norm = ds / np.sqrt(np.sum(ds**2, axis=1))[:,None]
    ds_norm = ds_norm.reshape((Sc+burn_in_critic, num_chains, D))
    # x0 = random.normal(subkeys[2], (num_chains, D))
    # return us, ds_norm, x0, key
    data_idx = random.randint(subkeys[2], (num_chains, ), 0, N)
    return us, ds_norm, data_idx, key

@jit
def generate_data_idx(key):
    key, subkey = random.split(key)
    data_idx = random.randint(subkey, (S, ), 0, N)
    return data_idx, key

# for ADAM
adam_iter = 0.0
m = np.zeros(len(theta))
v = np.zeros(len(theta))
adam_iter_phi = 0.0
m_phi = np.zeros(len(phi))
v_phi = np.zeros(len(phi))
b1 = 0.5
b2 = 0.9
step_size = 0.001
eps=10**-8

if dataset is "swiss":
    # d = np.load("wgan_swiss_v4_iter5000.npz")
    d = np.load("wgan_swiss_v4.npz")
elif dataset is "circle":
    d = np.load("wgan_circle.npz")
elif dataset is "mog":
    d = np.load("wgan_mog.npz")
m = d["m"]
v = d["v"]
m_phi = d["m_phi"]
v_phi = d["v_phi"]
adam_iter = d["adam_iter"]
adam_iter_phi = d["adam_iter_phi"]
theta = d["theta"]
phi = d["phi"]
losses = list(d["losses"])

@jit
def adam_step(theta, dL_dtheta, m, v, adam_iter):
    m = (1 - b1) * dL_dtheta      + b1 * m  # First  moment estimate.
    v = (1 - b2) * (dL_dtheta**2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1**(adam_iter + 1))            # Bias correction.
    vhat = v / (1 - b2**(adam_iter + 1))
    theta = theta - step_size*mhat/(np.sqrt(vhat) + eps)
    adam_iter = adam_iter + 1
    return theta, m, v, adam_iter

@jit 
def swap_axes(xs0, us, norm_ds, xLs, xRs, alphas):
    xs0 = np.swapaxes(xs0,0,1)
    us = np.swapaxes(us,0,1)
    norm_ds = np.swapaxes(norm_ds,0,1)
    xLs = np.swapaxes(xLs,0,1)
    xRs = np.swapaxes(xRs,0,1)
    alphas = np.swapaxes(alphas,0,1)
    return xs0, us, norm_ds, xLs, xRs, alphas

lamb_val = 10.0

n_critic = 10

# us, norm_ds, x0, key = generate_randomness_var(num_chains * n_critic, key)
# mu0, log_var0 = unflatten(theta)[-1]
# x0 = mu0 + np.sqrt(np.exp(log_var0)) * x0
us, norm_ds, data_idx, key = generate_randomness_var(num_chains * n_critic, key)
x0 = X[data_idx]
xs, _, _, _ = forwards(Sc+burn_in_critic, theta, x0, us, norm_ds)
xs = xs[burn_in_critic+1:].reshape((Sc*num_chains*n_critic, D))

data_idx, key = generate_data_idx(key)
ys = X[data_idx]

# forwards
# us, norm_ds, x0b, key = generate_randomness_burnin(key)
# data_idx, key = generate_data_idx(key)
# x0b = X[data_idx]
# # mu0, log_var0 = unflatten(theta)[-1]
# # x0b = mu0 + np.sqrt(np.exp(log_var0)) * x0b
# xs0, xLs, xRs, alphas = forwards(Sc+burn_in, theta, x0b, us, norm_ds)

# # xs = xs0[burn_in+1:].reshape((Sc*num_chains,D), order='F')
# xs = xs0[1:].reshape(( (Sc+burn_in) *num_chains,D), order='F') # use all but first sample

plt.figure()
plt.plot(X[:1000,0], X[:1000,1],'k.',label="true", alpha=0.5)
plt.plot(xs[:1000,0],xs[:1000,1],'r.',label="gen", alpha=0.5)
plt.legend()
plt.title("Iteration " + str(adam_iter))

plt.pause(0.1)

theta = d["theta"]
adam_iter=d["adam_iter"]
xmin =-2.0
xmax = 2.0
plt.figure()
visualize_2D(log_pdf, theta, xmin=xmin, xmax=xmax, vmin=0.0, vmax=0.4, dx=0.1, log=False, ax=plt.gca())
plt.title("Iteration " + str(adam_iter))
plt.plot(X[:500,0], X[:500,1], 'k.', markersize=2.5, alpha=0.5)


xmin = -2.0
xmax =2.0
plt.figure()
visualize_2D(discriminator, phi, xmin=xmin, xmax=xmax, vmin=0.0, vmax=0.25, dx=0.1, log=True, ax=plt.gca())
plt.title("Iteration " + str(adam_iter))
plt.plot(X[:500,0], X[:500,1], 'k.', markersize=2.5, alpha=0.5)
