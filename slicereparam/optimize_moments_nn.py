import autograd.numpy as np 
from autograd import grad
from autograd.misc import flatten
import numpy.random as npr
# npr.seed(0)

from scipy.optimize import root_scalar, brentq
import matplotlib.pyplot as plt
from tqdm.auto import trange
import sys

# def rbf_kernel(x, y, sigma=1.0):
#     """  
#     x and y are both N samples by D data points
#     """
#     N, D = x.shape
#     M, D = y.shape
#     pairwise_diffs = (x[None,:,:] - y[:,None,:]).reshape((N*M,D))
#     sqr_pairwise_diffs = np.sum(pairwise_diffs**2, axis=1)
#     out = np.exp( - 1.0 / 2.0 * sigma * sqr_pairwise_diffs)
#     return out # return sum? 

def rbf_kernel(x, y, sigma=1.0):
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
    x1 = np.arange(xmin,xmax+dx,dx)
    x2 = np.arange(xmin,xmax+dx,dx)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros(X1.shape)
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
    # ax.set_colorbar()
    # ax.tight_layout()

# Set up params
D = 2   # number of dimensions
S = 5  # number of samples
M = 2*D   # number of parameters

# parameters are mean and diagonal log covariance
H = 50
out_D = 1
# out_D = 2*D
W1 = 0.01 * npr.randn(H, D)
b1 = 0.01 * npr.randn(H)
W2 = 0.01 * npr.randn(H, H)
b2 = 0.01 * npr.randn(H)
W3 = 0.01 * npr.randn(out_D, H)
b3 = 0.001 * npr.randn(out_D)
_theta = [W1, b1, W2, b2, W3, b3, 0.0 * np.ones(D), 0.0 * np.ones(D)]

def _log_pdf(x, params):
    W1, b1 = params[:2]
    W2, b2 = params[2:4]
    W3, b3 = params[4:6]
    # h1 = np.maximum(W1@x + b1, 0.0)
    # h2 = np.maximum(W2@h1 + b2, 0.0)
    h1 = np.tanh(W1@x + b1)
    h2 = np.tanh(W2@h1 + b2)
    out = W3@h2 + b3
    mu = params[6]
    sigma_diag = np.exp(params[7])
    # return out + np.sum(-0.5 * (x **2 / 100)) # small regularization
    return out + np.sum(-0.5 * (x - mu) **2 / sigma_diag) 

# def _log_pdf(x, params):
#     W1, b1 = params[:2]
#     W2, b2 = params[2:]
#     # h1 = np.maximum(W1@x + b1, 0.0)
#     h1 = np.tanh(W1@x + b1)
#     out = W2@h1 + b2
#     mu = out[:D]
#     sigma_diag = np.exp(out[D:])
#     return np.sum(-0.5 * (x - mu) **2 / sigma_diag) # small regularization


def _gaussian_log_pdf(x, params):
    mu = params[0]
    sigma_diag = np.exp(params[1])
    return np.sum(-0.5 * (x - mu) **2 / sigma_diag)# - 0.5 *  np.log(np.linalg.det(np.diag(sigma_diag)))

theta, unflatten = flatten(_theta)
log_pdf = lambda x, params : _log_pdf(x, unflatten(params))
# log_pdf = lambda x, params : _log_pdf2(x, unflatten(params))


# Laplace
# b = 1.0
# def log_pdf(x, theta):
#     return -1.0 * np.sum( np.abs(x - theta) / b )

# compute necessary gradients
def log_pdf_theta(theta, x):    return log_pdf(x, theta)
def log_pdf_x(x, theta):        return log_pdf(x, theta)
def log_pdf_ad(x, theta, a, d): return log_pdf(x + a * d, theta)
grad_x = grad(log_pdf_x)
grad_theta = grad(log_pdf_theta)
grad_x_ad = grad(log_pdf_ad)

def f_alpha(alpha, x, d, theta, u1):
    return log_pdf(x + alpha * d, theta) - log_pdf(x, theta) - np.log(u1)

def forwards(S, theta, x, f_alpha, us, ds):
    xs = [x]
    xLs = []
    xRs = []
    alphas = []

    for s in range(S):

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

    return np.array(xs), xLs, xRs, alphas

# function for backwards pass
def backwards(S, theta, us, ds, xs, xLs, xRs, alphas,
              grad_theta, grad_x, grad_x_ad, dL_dxs,
              loss_grad_params, ys, bS=0):

    # if accept_samples is None:
        # accept_samples = np.ones(S)
    # Sdiv = np.sum(accept_samples) - bS

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

# sample random variables
us = npr.rand(S,2)
ds = npr.randn(S, D)
ds_norm = np.array([d / np.linalg.norm(d) for d in ds])
x = 0.1 * npr.randn(D) # initial x 


# ys = npr.randn(S, D)
xstar = 0.5 * npr.randn(D)

ys = xstar + npr.randn(S, D)

def features(xs):
    # D = xs.shape[1]
    S = xs.shape[0]
    x_mean = np.mean(xs,axis=0)
    x_demeaned = xs - x_mean
    x2_mean = np.mean(x_demeaned**2, axis=0) * S / (S-1)
    x1x2 = np.mean(x_demeaned[:,0] * x_demeaned[:,1]) * S / (S-1)
 
    return np.hstack((x_mean, x2_mean, x1x2))

# def _total_loss(xs, ys, params):
    # phi_x = features(xs)
    # phi_y = features(ys)
    # return np.sqrt(np.sum((phi_x - phi_y)**2))

def _total_loss(xs, ys, params, sigma=np.array([1.0,2.0,5.0,10.0])):
    k_xx = np.mean(rbf_kernel(xs, xs, sigma=sigma))
    k_xy = np.mean(rbf_kernel(xs, ys, sigma=sigma))
    k_yy = np.mean(rbf_kernel(ys, ys, sigma=sigma))
    return np.sqrt(k_xx - 2.0 * k_xy + k_yy)

loss = lambda x, y, params : _total_loss(x, y, unflatten(params))
total_loss = lambda xs, ys, params : _total_loss(xs, ys, unflatten(params))

# gradient of loss with respect to x
# loss_grad_x = grad(loss)
loss_grad_xs = grad(total_loss)
loss_grad_params = grad(lambda params, x, y : _total_loss(x, y, unflatten(params)))

def _loss_reparam(params, ds, y):
    xs = params[0] + np.sqrt(np.exp(params[1])) * ds
    # return np.sum(np.mean((xs - xstar)**2, axis=0)) - np.sum(0.5 * params[1])
    # x_mean = np.mean(xs, axis=0)
    # x2_mean = np.mean(xs**2, axis=0)
    # y_mean = np.mean(ys, axis=0)
    # y2_mean = np.mean(ys**2, axis=0)
    # return np.sum((x_mean-y_mean)**2) + 1.0 * np.sum((x2_mean-y2_mean)**2)
    phi_x = features(xs)
    phi_y = features(ys)
    return np.sqrt(np.sum((phi_x - phi_y)**2))
    # return np.mean((xs-y)**2) + np.mean((xs**2 - y**2)**2)
    # return np.sum((xs - xstar)**2) / xs.shape[0] - np.sum(0.5 * params[1])
loss_reparam = lambda params, ds, y : _loss_reparam(unflatten(params), ds, y)
grad_loss_reparam = grad(loss_reparam)

# compare gradients

# run forward pass
xs, xLs, xRs, alphas = forwards(S, theta, x, f_alpha, us, ds_norm)
# 2.0 * (np.mean(xs[1:],axis=0) - np.mean(ys, axis=0)) / S
# grad_v2 = 2.0 * (np.mean(xs[1:],axis=0) - np.mean(ys, axis=0)) / S \
        # + 2.0 * (np.mean(xs[1:]**2, axis=0) - np.mean(ys**2,axis=0)) * 2.0 * xs[1:] / S

# run backward pass
# loss_grad_x_i = lambda x, params : loss_grad_x(x, y, params)
# loss_grad_params_i = lambda params, x : loss_grad_params(params, x, y)
dL_dxs = loss_grad_xs(xs[1:], ys, theta)
dL_dtheta = backwards(S, theta, us, ds_norm, xs, xLs, xRs, alphas, 
                    grad_theta, grad_x, grad_x_ad, dL_dxs, loss_grad_params, ys)

print("Implicit: ", dL_dtheta)
# print("Explicit: ", grad_loss_reparam(theta, ds, ys))
# ys_new = theta[:D] + np.sqrt(np.exp(theta[D:])) * npr.randn(1000000, D)
# print("Explicit: ", grad_loss_reparam(theta, npr.randn(1000000, 2), ys_new))
# print("Explicit: ", grad_loss_reparam(theta, npr.randn(1000000,2), ys))
# exact_grad = np.concatenate(( 2.0 * (theta[:D] - xstar), np.exp(theta[D:]) - 0.5 )) 
# print("Exact   : ", exact_grad)

# compute gradient via finite differences
# dx = 1e-5
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

# # print("Implicit: ", dL_dtheta)
# print("Numerical: ", dthetas)
# print("MSE: ", np.mean((dL_dtheta - dthetas)**2)) 

# optimize parameters!
M = theta.shape[0]
# theta = np.hstack([0.5 * npr.randn(D), np.log(1.0*np.ones(D))])
# theta = np.hstack([0.5 * npr.randn(D), np.log(1.0 + 0.5 * npr.randn(D))])
# theta = 0.1 * npr.randn(M)
thetas = [theta]
xs = [x]
# losses = [loss(x, theta)]
losses = [0.0]
S = 64
num_iters=250

# learning rate params
a0 = 0.01
gam = 0.0

# plt.figure()
# plt.plot(np.arange(num_iters), a0 / (1 + gam * (np.arange(num_iters)+1)))

# TRUE VARIANCE
# true_var = 1.0
# true_var = np.exp(0.2 * npr.randn(D))
Cov = np.array([[1.25, 0.5],[0.5,1.0]])
Chol = np.linalg.cholesky(Cov)

# ys = xstar + npr.randn(100000, D) @ Chol.T
# print("Features: ", features(ys))

# for adam
m = np.zeros(len(theta))
v = np.zeros(len(theta))
b1 = 0.5
b2 = 0.9
step_size = 0.01
eps=10**-8

pbar = trange(num_iters)
pbar.set_description("Loss: {:.1f}".format(losses[0]))
for i in range(num_iters):

    us = npr.rand(S, 2)
    ds = npr.randn(S, D)
    norm_ds = np.array([d / np.linalg.norm(d) for d in ds])
    # ys = xstar + np.sqrt(true_var) * npr.randn(1000, D)
    ys = xstar + npr.randn(S, D) @ Chol.T

    # forward pass
    x0 = xs[-1]
    # x0 = theta[:D] + np.sqrt(np.exp(theta[D:])) * npr.randn(D)
    xs, xLs, xRs, alphas = forwards(S, theta, x0, f_alpha, us, norm_ds)

    # backwards pass
    dL_dxs = loss_grad_xs(xs[1:], ys, theta)
    dL_dtheta = backwards(S, theta, us, norm_ds, xs, xLs, xRs, alphas,
                      grad_theta, grad_x, grad_x_ad, dL_dxs, loss_grad_params, ys, bS=0)
    #dL_dtheta += loss_grad_params(theta, xs[1:])

    # update parameters

    # SGD
    # alpha_t = a0 / (1 + gam * (i+1)) # learning rate 
    # theta = theta - dL_dtheta * alpha_t

    # ADAM
    m = (1 - b1) * dL_dtheta      + b1 * m  # First  moment estimate.
    v = (1 - b2) * (dL_dtheta**2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1**(i + 1))            # Bias correction.
    vhat = v / (1 - b2**(i + 1))
    theta = theta - step_size*mhat/(np.sqrt(vhat) + eps)

    thetas.append(theta)
    losses.append(loss(xs[1:], ys, theta))

    pbar.set_description("Loss: {:.1f}".format(losses[-1]))
    pbar.update()

pbar.close()

# thetas_plot = np.array(thetas)

# plt.savefig("optimize_moments_dim" + str(D) + "_samples" + str(S) + ".png")

gauss_log_pdf = lambda x : -0.5 * (x - xstar).T @ np.linalg.inv(Cov) @ (x - xstar)

xmin = -5.0
xmax =5.0
# plt.figure()
# # plt.subplot(121)
# plt.hist2d(ys[:,0], ys[:,1], range=[[xmin, xmax],[xmin, xmax]], bins=50, density=True)#, vmin=0.0, vmax=0.1)
# plt.xlim([xmin, xmax])
# plt.ylim([xmin, xmax])
# plt.colorbar()
plt.figure()
ax1 = plt.subplot(121)
visualize_2D(gauss_log_pdf, xmin=xmin, xmax=xmax, ax=ax1, vmin=0.0, vmax=0.17, dx=0.1)
plt.title("True")
ax2 = plt.subplot(122)
visualize_2D(log_pdf, theta, xmin=xmin, xmax=xmax, ax=ax2, vmin=0.0, vmax=0.17, dx=0.1)
plt.title("Generative")


# test sample a lot of xs
S2 = 1000
us = npr.rand(S2, 2)
ds = npr.randn(S2, D)
norm_ds = np.array([d / np.linalg.norm(d) for d in ds])
x0 = xs[-1]
xs2, xLs, xRs, alphas = forwards(S2, theta, x0, f_alpha, us, norm_ds)
