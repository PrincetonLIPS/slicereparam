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

@jit
def fa(x, alpha, d, theta, u1):
    return log_pdf(x + alpha * d, theta) - log_pdf(x, theta) - np.log(u1)

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

a_grid = np.concatenate((np.logspace(-3,1, 25), np.array([25.0])))

@jit
def fa_grid(x, d, theta, u1):
    fout = []
    for a in a_grid:
        fout.append(f_alpha(a, x, d, theta, u1))
    return np.array(fout)

@jit
def fma_grid(x, d, theta, u1):
    fout = []
    for a in a_grid:
        fout.append(f_alpha(-1.0 * a, x, d, theta, u1))
    return np.array(fout)

# @jit
@jit
def forwards_step(x, theta, u1, u2, d, aL, bR):
    # z_L = bisect_method(x, d, theta, u1, a=-25.0, b=-1e-10)
    # z_R = bisect_method(x, d, theta, u1, a=1e-10, b=25.0)
    # z_L, z_R = dual_bisect_method(x, d, theta, u1, aL=-25.0, bL=-1e-10, aR=1e-10, bR=25.0)
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
        # aL=a_grid[np.where(fa(x, -a_grid, ds[s], theta, us[s,0])<0)[0][0]]*-1.0
        # aL=a_grid[np.where(fa(x, -a_grid, ds[s], theta, us[s,0])<0)[0][0]]*-1.0
        aL=a_grid[np.where(fa_grid(x, ds[s], theta, us[s,0])<0)[0][0]]*-1.0
        bR=a_grid[np.where(fma_grid(x, ds[s], theta, us[s,0])<0)[0][0]]
        x, x_L, x_R, alpha = forwards_step(x, theta, us[s,0], us[s,1], ds[s], aL, bR)
        xs.append(x)
        xLs.append(x_L)
        xRs.append(x_R)
        alphas.append(alpha)
    return np.array(xs), np.array(xLs), np.array(xRs), np.array(alphas)

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
key = random.PRNGKey(3)

# Set up params
D = 2   # number of latent dimensions
D_out = 784 # dimensionality of data
# H_energy = 25
H_model = 200
scale = 0.001

# energy_layer_sizes = [D, H_energy, H_energy, 1]
decoder_layer_sizes = [D_out, H_model, H_model, 1]

key, subkey = random.split(key)
_decoder_params, key = init_random_params(scale, decoder_layer_sizes, subkey)
_decoder_params += [[0.0 * np.ones(D), 0.0 * np.ones(D)]] # gaussian normalizer
# _decoder_params += [[0.0 * np.ones(D_out), 0.0 * np.ones(D_out)]] # gaussian normalizer

@jit
def _decoder(x, params):
    inputs = x
    generate_params, decoder_params = params
    decoder_nn_params = decoder_params[:-1]
    mu, log_sigma_diag = decoder_params[-1]
    sigma_diag = np.exp(log_sigma_diag)
    for W, b in decoder_nn_params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)     
    outW, outb = decoder_nn_params[-1]
    out = np.dot(inputs, outW) + outb
    return out 

def _log_pdf(x, params):
    generate_params, decoder_params = params

    inputs = x
    for W, b in generate_params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)                            # nonlinear transformation
    outW, outb = generate_params[-1]
    output = sigmoid(np.dot(inputs, outW) + outb)

    # discrimator / decoder?
    inputs = output
    decoder_nn_params = decoder_params[:-1]
    mu, log_sigma_diag = decoder_params[-1]
    sigma_diag = np.exp(log_sigma_diag)
    for W, b in decoder_nn_params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)     

    outW, outb = decoder_nn_params[-1]
    out = np.dot(inputs, outW) + outb

    return np.sum(out) + np.sum(-0.5 * (x - mu) **2 / sigma_diag)

model_layer_sizes = [D, H_model, H_model, D_out]
key, subkey = random.split(key)
_generate_params, key = init_random_params(scale, model_layer_sizes, subkey)

def _generate(xs, params):
    generate_params, decoder_params = params
    inputs = xs
    for W, b in generate_params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)                            # nonlinear transformation
    outW, outb = generate_params[-1]
    outputs = sigmoid(np.dot(inputs, outW) + outb)
    return outputs

_params = [_generate_params, _decoder_params]
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
    k_yy = np.mean(rbf_kernel(ys, ys, sigma=sigma))
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


from jax import lax

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

# test functions
S = 64 # number of samples
key, *subkeys = random.split(key, 4)
us = random.uniform(subkeys[0], (S,2))
ds = random.normal(subkeys[1], (S,D))
ds_norm = np.array([d / np.linalg.norm(d) for d in ds])
x = 0.1 * random.normal(subkeys[2], (D,)) # initial x 

# run forward pass
xs, xLs, xRs, alphas = forwards(S, params, x, us, ds_norm)

# run backward pass
key, *subkeys = random.split(key)
ys = random.normal(key, (S, D_out))
dL_dxs = loss_grad_xs(xs[1:], ys, params)
dL_dtheta2 = backwards3(S, params, us, ds_norm, xs, xLs, xRs, alphas, dL_dxs, ys)
print("Implicit: ", dL_dtheta2)

# load data
N, train_images, _, test_images, _ = load_mnist()

# # optimize parameters!
# d = np.load("mmdgan_weights_64_v4.npz")
# theta = d["theta"]
# m=d["m"]
# v=d["v"]
# adam_iter=d["adam_iter"]

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
    key, *subkeys = random.split(key, 4)
    us = random.uniform(subkeys[0], (S,2))
    ds = random.normal(subkeys[1], (S,D))
    ds_norm = np.array([d / np.linalg.norm(d) for d in ds])
    data_idx = random.randint(key, (S, ), 0, N)
    x0 = random.normal(subkeys[2], (D,))
    return us, ds_norm, data_idx, x0, key

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

import time
t1 = time.time()
fig = plt.figure(figsize=[8,6])
num_epochs = 1
#pbar = trange(num_iters*num_epochs)
#pbar.set_description("Loss: {:.1f}".format(losses[0]))
for epoch in range(num_epochs):
    for i in range(num_iters):

        us, norm_ds, data_idx, x0, key = generate_randomness(key)

        ys = train_images[data_idx]

        # forward pass
        # x0 = xs[-1]
        xs, xLs, xRs, alphas = forwards(S, theta, x0, us, norm_ds)

        # backwards pass
        dL_dxs = loss_grad_xs(xs[1:], ys, theta)
        # dL_dtheta = backwards(S, theta, us, norm_ds, xs, xLs, xRs, alphas, grad_theta, grad_x, grad_x_ad, dL_dxs, loss_grad_params, ys)
        dL_dtheta = backwards3(S, theta, us, norm_ds, xs, xLs, xRs, alphas, dL_dxs, ys)

        # ADAM
        theta, m, v, adam_iter = adam_step(theta, dL_dtheta, m, v, adam_iter)

    #    if np.mod(i, 10) == 0:
    #        key = plot_update(xs, theta, key)
    #         thetas.append(theta)

        losses.append(total_loss(xs[1:], ys, theta))
        if np.mod(i,10)==0:
            key = plot_update(xs, theta, key)
            t2=time.time()
            print("Epoch: ", epoch, "Iter: ", i, "Loss: ", losses[-1], "Time: ", t2-t1)

        # if np.mod(i,250)==0:
            # np.savez("mmdgan_weights_64_v4.npz", theta=theta, losses=np.array(losses), num_epochs=num_epochs, m=m, v=v, adam_iter=adam_iter)

# np.savez("mmdgan_weights_64_v4.npz", theta=theta, losses=np.array(losses), num_epochs=num_epochs, m=m, v=v, adam_iter=adam_iter)

# investigate learned network
key, *subkeys = random.split(key, 4)
train_idx = random.randint(subkeys[0], (S, ), 0, N)
test_idx = random.randint(subkeys[1], (S, ), 0, len(test_images))

train_energies = _decoder(train_images[train_idx], unflatten(theta))
test_energies = _decoder(test_images[test_idx], unflatten(theta))
# oos_input = 0.5 + random.normal(subkeys[2], (S, D_out))
oos_input = 0.5 + 0.1 * random.normal(subkeys[2], (S, D_out))
ood_energies = _decoder(oos_input, unflatten(theta))

plt.figure()
plt.plot(train_energies, label="train")
plt.plot(test_energies, label="test")
plt.plot(ood_energies, label="ood")
plt.legend()

key, subkey = random.split(key)
z0 = random.normal(subkey, (D, ))
test_image = oos_input[0]

# x_images = _generate(xs[rand_idx], unflatten(theta))
def loss_z0(z0, x_image):
    x0 = _generate(z0, unflatten(theta))
    return np.mean((x0 - x_image)**2)
grad_z0 = jit(grad(loss_z0))
alpha = 1.0
for i in range(1000):
    z0_g = grad_z0(z0, test_image)
    z0 = z0 - z0_g * alpha

plt.figure()
plt.subplot(121)
plt.imshow(_generate(z0, unflatten(theta)).reshape((28,28)), cmap="gray", vmin=0, vmax=1)
plt.subplot(122)
plt.imshow(test_image.reshape((28,28)), cmap="gray", vmin=0, vmax=1)