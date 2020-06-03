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

gen_layer_sizes = [D, H_energy, H_energy, D_out]
key, subkey = random.split(key)
_gen_params, key = init_random_params(scale, gen_layer_sizes, subkey)

def _generator(x, params):
    inputs = x
    for W, b in params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)   
    outW, outb = params[-1]
    out = np.dot(inputs, outW)+ outb
    return out

gen_params, unflatten_gen = ravel_pytree(_gen_params)
generator = jit(lambda x, params : _generator(x, unflatten_gen(params)))

disc_layer_sizes = [D, H_energy, H_energy, 1]
key, subkey = random.split(key)
_disc_params, key = init_random_params(scale, disc_layer_sizes, subkey)

def _discriminator(x, params):
    inputs = x
    for W, b in params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)   
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
def gen_loss(gen_params, disc_params, zs):
    xs = generator(zs, gen_params)
    return - 1.0 * np.mean(discriminator(xs, disc_params))

# gradient of loss with respect to x
gen_loss_grad = jit(grad(gen_loss))

# load data
from sklearn.datasets import make_swiss_roll
N = 100000
noise = 0.25
X, _ = make_swiss_roll(N, noise)
X /= 7.5
X = X[:,[0,2]]

# # optimize parameters!
theta = gen_params+0.0
phi = disc_params + 0.0
# d = np.load("wgan_swiss.npz")
# theta = d["theta"]
# phi = d["phi"]
M = theta.shape[0]
losses = [0.0]

# set up iters
S = 256
num_iters = int(np.ceil(N / S))
num_chains = 10
Sc = int(S / num_chains)

@jit
def batch_indices(iter):
    idx = iter % num_iters
    return slice(idx * S, (idx+1) * S) # S is batch size

@jit
def generate_randomness(key):
    key, *subkeys = random.split(key, 3)
    zs = random.normal(subkeys[0], (S, D))
    data_idx = random.randint(subkeys[1], (S, ), 0, N)
    return zs, data_idx, key

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

n_critic = 10
lamb_val = 10.0

fig = plt.figure(figsize=[8,6])
num_epochs = 5
pbar = trange(num_iters*num_epochs)
pbar.set_description("Loss: {:.1f}".format(losses[0]))
import time 
t1 = time.time()
for epoch in range(num_epochs):
    for i in range(num_iters):

        for k in range(n_critic):
            zs, data_idx, key = generate_randomness(key)
            key, subkey = random.split(subkey)
            es = random.uniform(subkey, (S,1))
            ys = X[data_idx]
            xs = generator(zs, theta)
            xhats = ys * es + xs * (1.0 - es)

            grad_phi = grad_L_func(phi, xs, ys, xhats, lamb_val)
            phi, m_phi, v_phi, adam_iter_phi = adam_step(phi, grad_phi, m_phi, v_phi, adam_iter_phi)


        # forwards
        zs, data_idx, key = generate_randomness(key)
        dL_dtheta = gen_loss_grad(theta, phi, zs)

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

        xs = generator(zs, theta)
        losses.append(L_func(phi, xs, ys, xhats, lamb_val))# + gen_loss(xs[1:], phi))

        pbar.set_description("Loss: {:.1f}".format(losses[-1]))
        pbar.update()

# np.savez("wgan_swiss.npz", theta=theta, phi=phi, losses=np.array(losses), \
    # m=m, v=v, adam_iter=adam_iter, m_phi=m_phi, v_phi=v_phi, adam_iter_phi=adam_iter_phi)
# pbar.close()

# thetas_plot = np.array(thetas)

# plt.savefig("optimize_moments_dim" + str(D) + "_samples" + str(S) + ".png")

# gauss_log_pdf = lambda x : -0.5 * (x - xstar).T @ np.linalg.inv(Cov) @ (x - xstar)

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

# xmin = -3.0
# xmax =3.0
# plt.figure()
# ax1=plt.subplot(121)
# visualize_2D(log_pdf, params, xmin=xmin, xmax=xmax, vmin=0.0, vmax=0.15, dx=0.1,ax=ax1)
# plt.plot(X[:2500,0], X[:2500,1], 'r.', markersize=2.5)
# plt.title("Init")
# ax2=plt.subplot(122)
# visualize_2D(log_pdf, theta, xmin=xmin, xmax=xmax, vmin=0.0, vmax=0.15, dx=0.1, ax =ax2)
# plt.plot(X[:2500,0], X[:2500,1], 'r.', markersize=2.5)
# plt.title("Generative, Iteration: " + str(len(losses)))

# d = np.load("wgan_swiss.npz")
# theta = d["theta"]
# adam_iter = d["adam_iter"]
xmin = -3.0
xmax =3.0
plt.figure()
visualize_2D(log_pdf, theta, xmin=xmin, xmax=xmax, vmin=0.0, vmax=0.15, dx=0.1)
plt.title("Generative, Iteration: " + str(adam_iter))
plt.plot(X[:500,0], X[:500,1], 'r.', markersize=2.5)


# plt.figure()
# plt.subplot(221)
# plt.plot(xs[:,0], xs[:,1])
# test sample a lot of xs
# S2 = 1000
# us = npr.rand(S2, 2)
# ds = npr.randn(S2, D)
# norm_ds = np.array([d / np.linalg.norm(d) for d in ds])
# x0 = xs[-1]
# xs2, xLs, xRs, alphas = forwards(S2, theta, x0, f_alpha, us, norm_ds)
# idx=-1
# images = _generate(xs_new, unflatten(theta))
# idx+=1
# plt.figure()
# plt.imshow(images[idx].reshape((28,28)), cmap="gray", vmin=0.0, vmax=1.0)

# key, subkey = random.split(key)
# # jit_generate=jit(_generate)
# images = _generate(random.normal(subkey, (1, D)), unflatten(theta))
# # images = jit_generate(random.normal(subkey, (1, D)), unflatten(theta))
# plt.figure()
# plt.imshow(images[0].reshape((28,28)), cmap="gray", vmin=0.0, vmax=1.0)