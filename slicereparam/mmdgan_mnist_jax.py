from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np 
from jax import jit, grad
from jax import random
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
    ax.imshow(np.exp(Z) / (np.sum(np.exp(Z)) * dx**2), extent=[xmin,xmax,xmin,xmax], origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([xmin, xmax])

@jit
def f_alpha(alpha, x, d, theta, u1):
    return log_pdf(x + alpha * d, theta) - log_pdf(x, theta) - np.log(u1)

def forwards(S, theta, x, f_alpha, us, ds):
    xs = [x]
    xLs = []
    xRs = []
    alphas = []

    for s in range(S):

        # import ipdb; ipdb.set_trace()
        u1 = us[s,0]
        u2 = us[s,1]
        d = ds[s]

        fz = lambda alpha : f_alpha(alpha, x, d, theta, u1)
        z_L = brentq(fz, a=-1e8, b=-1e-10)
        z_R = brentq(fz, a=1e-10, b=1e8)
        # fz = lambda alpha : f_alpha(alpha, x, d, theta, u1)
        # z_L = brentq(f_alpha, args=(x, d, theta, u1), a=-1e8, b=-1e-10)
        # z_R = brentq(f_alpha, args=(x, d, theta, u1), a=1e-10, b=1e8)
        x_L = x + d*z_L
        x_R = x + d*z_R
        x = (1 - u2) * x_L + u2 * x_R

        xs.append(x)
        xLs.append(x_L)
        xRs.append(x_R)
        alphas.append(np.array([z_L,z_R]))

    return np.array(xs), np.array(xLs), np.array(xRs), np.array(alphas)


S=64
@partial(jit, static_argnums=(0,8,9,10,12))
def backwards(S, theta, us, ds, xs, xLs, xRs, alphas,
              grad_theta, grad_x, grad_x_ad, dL_dxs,
              loss_grad_params, ys):

    D = xs[0].shape[0]
    dL_dtheta = np.zeros_like(theta)
    for s in range(S-1, -1, -1):

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
D_out = 784 # dimensionality of data
H_energy = 25
H_model = 100
scale = 0.001

# energy_layer_sizes = [D, H_energy, H_energy, 1]
energy_layer_sizes = [D, H_energy, 1]

key, subkey = random.split(key)
_energy_params, key = init_random_params(scale, energy_layer_sizes, subkey)
_energy_params += [[0.0 * np.ones(D), 0.0 * np.ones(D)]] # gaussian normalizer

def _log_pdf(x, params):
    energy_params, model_params = params
    nn_params = energy_params[:-1]
    mu, log_sigma_diag = energy_params[-1]
    sigma_diag = np.exp(log_sigma_diag)
    inputs = x
    for W, b in nn_params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)   
        # inputs = np.tanh(outputs)   

    outW, outb = nn_params[-1]
    out = np.dot(inputs, outW)+ outb
    return np.sum(out) + np.sum(-0.5 * (x - mu) **2 / sigma_diag) 

model_layer_sizes = [D, H_model, H_model*2, D_out]
key, subkey = random.split(key)
_model_params, key = init_random_params(scale, model_layer_sizes, subkey)

def _generate(xs, params):
    energy_params, model_params = params
    inputs = xs
    for W, b in model_params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)                            # nonlinear transformation
    outW, outb = model_params[-1]
    outputs = sigmoid(np.dot(inputs, outW) + outb)
    return outputs

_params = [_energy_params, _model_params]
params, unflatten = ravel_pytree(_params)
log_pdf = jit(lambda x, params : _log_pdf(x, unflatten(params)))

# compute necessary gradients
def log_pdf_theta(theta, x):    return log_pdf(x, theta)
def log_pdf_x(x, theta):        return log_pdf(x, theta)
def log_pdf_ad(x, theta, a, d): return log_pdf(x + a * d, theta)
grad_x = jit(grad(log_pdf_x))
grad_theta = jit(grad(log_pdf_theta))
grad_x_ad = jit(grad(log_pdf_ad))

def _total_loss(xs, ys, params, sigma=np.array([1.0,2.0,5.0,10.0,20.0,50.0])):
    outs = _generate(xs, params)
    k_xx = np.mean(rbf_kernel(outs, outs, sigma=sigma))
    k_xy = np.mean(rbf_kernel(outs, ys, sigma=sigma))
    k_yy = np.mean(rbf_kernel(outs, ys, sigma=sigma))
    return np.sqrt(k_xx - 2.0 * k_xy + k_yy)

loss = lambda x, y, params : _total_loss(x, y, unflatten(params))
total_loss = jit(lambda xs, ys, params : _total_loss(xs, ys, unflatten(params)))

# gradient of loss with respect to x
# loss_grad_x = grad(loss)
loss_grad_xs = jit(grad(total_loss))
loss_grad_params = jit(grad(lambda params, x, y : _total_loss(x, y, unflatten(params))))

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
    R_grad_theta = -1.0 * (grad_theta(theta, xL) - grad_theta(theta, x)) / np.dot(d, grad_x_ad(x, theta, z_R, d))

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

from jax import lax
# def backwards3(S, theta, us, ds, xs, xLs, xRs, alphas, dL_dxs, ys):
#     dL_dtheta = np.zeros_like(theta)
#     prev_dL_dx = np.zeros_like(xs[0])
#     def body_fun(i, val):
#         import ipdb; ipdb.set_trace()
#         s = i.astype(int)
#         dL_dtheta, prev_dL_dx = val 
#         dL_dtheta, prev_dL_dx = backwards_step(theta, dL_dtheta, us[s,:], ds[s], xs[s], 
#                                                xLs[s], xRs[s], alphas[s], dL_dxs[s], prev_dL_dx)
#         return [dL_dtheta, prev_dL_dx]
#     lax.fori_loop(-(S-1), 0, body_fun, [dL_dtheta, prev_dL_dx])
#     return dL_dtheta + loss_grad_params(theta, xs[1:], ys)
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
    return dL_dtheta + loss_grad_params(theta, xs[1:], ys)

# test_array = np.arange(S)
# def cond_fun(val):
#     return val[0] > -1

# def body_fun(val):
#     # val[1] += val[0]
#     val[1] += test_array[val[0]]
#     val[0] -= 1
#     return val 

# out = lax.while_loop(cond_fun, body_fun, [S-1, 0])

# test functions
S = 64 # number of samples
key, *subkeys = random.split(key, 4)
us = random.uniform(subkeys[0], (S,2))
ds = random.normal(subkeys[1], (S,D))
ds_norm = np.array([d / np.linalg.norm(d) for d in ds])
x = 0.1 * random.normal(subkeys[2], (D,)) # initial x 

# run forward pass
xs, xLs, xRs, alphas = forwards(S, params, x, f_alpha, us, ds_norm)

# run backward pass
key, *subkeys = random.split(key)
ys = random.normal(key, (S, D_out))
dL_dxs = loss_grad_xs(xs[1:], ys, params)
# dL_dtheta = backwards(S, params, us, ds_norm, xs, xLs, xRs, alphas, grad_theta, grad_x, grad_x_ad, dL_dxs, loss_grad_params, ys)
dL_dtheta = backwards2(S, params, us, ds_norm, xs, xLs, xRs, alphas, dL_dxs, ys)
dL_dtheta2 = backwards3(S, params, us, ds_norm, xs, xLs, xRs, alphas, dL_dxs, ys)
# t1 = time.time(); dL_dtheta2 = backwards3(S, params, us, ds_norm, xs, xLs, xRs, alphas, dL_dxs, ys); t2 = time.time(); print(t2-t1)
print("Implicit: ", dL_dtheta)

# load data
N, train_images, _, test_images, _ = load_mnist()

# # optimize parameters!
theta = params+0.0
M = theta.shape[0]
thetas = [theta]
xs = [x]
losses = [0.0]

# set up iters
S = 64
num_iters = int(np.ceil(len(train_images) / S))

@jit
def batch_indices(iter):
    idx = iter % num_iters
    return slice(idx * S, (idx+1) * S) # S is batch size

# set up randomness
@jit
def generate_randomness(key):
    key, *subkeys = random.split(key, 3)
    us = random.uniform(subkeys[0], (S,2))
    ds = random.normal(subkeys[1], (S,D))
    ds_norm = np.array([d / np.linalg.norm(d) for d in ds])
    data_idx = random.randint(key, (S, ), 0, N)
    return us, ds_norm, data_idx, key

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
    x_images = _generate(xs[rand_idx], unflatten(theta))
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
for epoch in range(num_epochs):
    for i in range(num_iters):

        us, norm_ds, data_idx, key = generate_randomness(key)

        ys = train_images[data_idx]

        # forward pass
        x0 = xs[-1]
        xs, xLs, xRs, alphas = forwards(S, theta, x0, f_alpha, us, norm_ds)

        # backwards pass
        dL_dxs = loss_grad_xs(xs[1:], ys, theta)
        # dL_dtheta = backwards(S, theta, us, norm_ds, xs, xLs, xRs, alphas, grad_theta, grad_x, grad_x_ad, dL_dxs, loss_grad_params, ys)
        dL_dtheta = backwards3(S, theta, us, norm_ds, xs, xLs, xRs, alphas, dL_dxs, ys)

        # ADAM
        theta, m, v, adam_iter = adam_step(theta, dL_dtheta, m, v, adam_iter)

        if np.mod(i, 25) == 0:
            key = plot_update(xs, theta, key)
            # thetas.append(theta)

        losses.append(total_loss(xs[1:], ys, theta))

        pbar.set_description("Loss: {:.1f}".format(losses[-1]))
        pbar.update()

pbar.close()

# thetas_plot = np.array(thetas)

# plt.savefig("optimize_moments_dim" + str(D) + "_samples" + str(S) + ".png")

# gauss_log_pdf = lambda x : -0.5 * (x - xstar).T @ np.linalg.inv(Cov) @ (x - xstar)

xmin = -5.0
xmax =5.0
plt.figure()
visualize_2D(log_pdf, theta, xmin=xmin, xmax=xmax, vmin=0.0, vmax=0.17, dx=0.1)
plt.title("Generative")

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