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
from jax.scipy.special import expit as sigmoid

def batch_normalize(activations):
    mbmean = np.mean(activations, axis=0, keepdims=True)
    return (activations - mbmean) / (np.std(activations, axis=0, keepdims=True) + 1)

def relu(x):       
    return np.maximum(0, x)

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
key = random.PRNGKey(4413)

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
from sklearn.datasets import make_swiss_roll
N = 100000
noise = 0.25
X, _ = make_swiss_roll(N, noise)
X /= 7.5
X = X[:,[0,2]]

# N = 100000
# noise = 0.13
# mus = np.array([np.array([1.0,1.0]), np.array([-1.0,1.0]), np.array([-1.0,-1.0]), np.array([1.0,-1.0])])
# key, subkey = random.split(key)
# zs = random.randint(key, (N, ), minval=0, maxval=4)
# mu_zs = mus[zs]
# key, subkey = random.split(key)
# X = mu_zs + np.sqrt(noise) * random.normal(key, (N, D))

# N = 100000
# noise = 0.001
# rs = np.array([0.5,1.0,1.5,2.0])
# key, subkey = random.split(key)
# zs = random.randint(subkey, (N, ), minval=0, maxval=4)
# r_zs = rs[zs]
# key, subkey = random.split(key)
# angles = 2.0 * np.pi * random.uniform(subkey, (N,))
# X = r_zs[:,None] * np.hstack((np.cos(angles)[:,None], np.sin(angles[:,None])))
# key, subkey = random.split(key)
# X = X + np.sqrt(noise) * random.normal(subkey, (N, D))
# plt.figure()
# plt.plot(X[:1000,0], X[:1000,1],'k.')

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
    us = random.uniform(subkeys[0], (burn_in,num_chains,2))
    ds = random.normal(subkeys[1], (burn_in*num_chains,D))
    ds_norm = ds / np.sqrt(np.sum(ds**2, axis=1))[:,None]
    ds_norm = ds_norm.reshape((burn_in, num_chains, D))
    x0 = random.normal(subkeys[2], (num_chains, D))
    return us, ds_norm, x0, key

burn_in_critic = 2
@partial(jit, static_argnums={0})
def generate_randomness_var(num_chains, key):
    S_total = (Sc+burn_in_critic) * num_chains
    key, *subkeys = random.split(key, 4)
    us = random.uniform(subkeys[0], (Sc+burn_in_critic,num_chains,2))
    ds = random.normal(subkeys[1], (S_total,D))
    ds_norm = ds / np.sqrt(np.sum(ds**2, axis=1))[:,None]
    ds_norm = ds_norm.reshape((Sc+burn_in_critic, num_chains, D))
    x0 = random.normal(subkeys[2], (num_chains, D))
    return us, ds_norm, x0, key

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
step_size = 0.0001
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

num_epochs = 50
import time 
t1 = time.time()
for epoch in range(num_epochs):
    for i in range(num_iters):

        if epoch == 0 and i < 25:
            n_critic = 100
        else:
            n_critic = 10 

        us, norm_ds, x0, key = generate_randomness_var(num_chains * n_critic, key)
        mu0, log_var0 = unflatten(theta)[-1]
        x0 = mu0 + np.sqrt(np.exp(log_var0)) * x0
        xs, _, _, _ = forwards(Sc+burn_in_critic, theta, x0, us, norm_ds)
        # xs = xs[1:].reshape((Sc*num_chains*n_critic, D))
        xs = xs[-1].reshape((Sc*num_chains*n_critic, D))

        for k in range(n_critic):
            # us, norm_ds, data_idx, x0, key = generate_randomness(key)
            key, subkey = random.split(key)
            es = random.uniform(subkey, (S,1))

            data_idx, key = generate_data_idx(key)
            ys = X[data_idx]

            # xs, _, _, _ = forwards(Sc, theta, x0, us, norm_ds)
            # xs = xs[1:].reshape((Sc*num_chains, D))
            xhats = ys * es + xs[k*S:(k+1)*S] * (1.0 - es)

            grad_phi = grad_L_func(phi, xs[k*S:(k+1)*S], ys, xhats, lamb_val)
            phi, m_phi, v_phi, adam_iter_phi = adam_step(phi, grad_phi, m_phi, v_phi, adam_iter_phi)

        # forwards
        us, norm_ds, x0b, key = generate_randomness_burnin(key)
        mu0, log_var0 = unflatten(theta)[-1]
        x0b = mu0 + np.sqrt(np.exp(log_var0)) * x0b
        xs0, xLs, xRs, alphas = forwards(Sc+burn_in, theta, x0b, us, norm_ds)
        xs = xs0[burn_in+1:].reshape((Sc*num_chains,D), order='F')

        # backwards pass
        dL_dxs = loss_grad_xs(xs, phi)
        dL_dxs_temp = dL_dxs + 0.0
        dL_dxs = dL_dxs.reshape((num_chains, Sc, D))
        dL_dxs = np.hstack((np.zeros((num_chains,burn_in,D)), dL_dxs))

        xs0, us, norm_ds, xLs, xRs, alphas = swap_axes(xs0, us, norm_ds, xLs, xRs, alphas)
        dL_dthetas = vmapped_backwards(Sc, theta, us, norm_ds, xs0, xLs, xRs, alphas, dL_dxs, ys)
        dL_dtheta = np.sum(dL_dthetas,axis=0)

        theta, m, v, adam_iter = adam_step(theta, dL_dtheta, m, v, adam_iter)

        losses.append(L_func(phi, xs, ys, xhats, lamb_val) + total_loss(xs[1:], phi))

        if np.mod(i,50)==0:
            t2=time.time()
            print("Epoch: ", epoch, "Iter: ", i, "Loss: ", losses[-1], "Time: ", t2-t1)
            t1=time.time()

        if np.mod(i,100)==0:
            np.savez("wgan_swiss_v2.npz", theta=theta, phi=phi, losses=np.array(losses), \
                m=m, v=v, adam_iter=adam_iter, m_phi=m_phi, v_phi=v_phi, adam_iter_phi=adam_iter_phi, \
                energy_layer_sizes=energy_layer_sizes, disc_layer_sizes=disc_layer_sizes)

np.savez("wgan_swiss_v2.npz", theta=theta, phi=phi, losses=np.array(losses), \
    m=m, v=v, adam_iter=adam_iter, m_phi=m_phi, v_phi=v_phi, adam_iter_phi=adam_iter_phi, \
    energy_layer_sizes=energy_layer_sizes, disc_layer_sizes=disc_layer_sizes)





