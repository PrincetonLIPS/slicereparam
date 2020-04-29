import autograd.numpy as np 
from autograd import grad
# import jax.numpy as np 
# from jax import grad

import numpy.random as npr
npr.seed(0)
# import numpy as onp 
from scipy.optimize import root_scalar, brentq
import matplotlib.pyplot as plt
from tqdm.auto import trange

# Set up params
D = 3   # number of dimensions
S = 5   # number of samples
M = 3   # number of parameters

theta = 0.0 * np.ones(D)     # parameters
# theta = 0.1 * np.ones((D,1000))     # parameters

# log_pdf : function that returns log pi_\theta(x) - this function can be unnormalized 

# Gaussian 
Sigma = np.eye(D)                           # identity covariance
# L = npr.randn(3,3)*0.5; Sigma = L@L.T     # random covariance

def log_pdf(x, theta):
    mu = theta
    # mu = np.sum(theta,axis=1)
    return -0.5 * (x - mu).T @ np.linalg.inv(Sigma) @ (x - mu)

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

# jit_brentq = jit(partial(brentq, a=-1e8, b=-1e-12))
# jit_brentq_l = jit(lambda fz : brentq(fz, a=-1e8, b=-1e-12), static_argnums=(0))
# jit_brentq_r = jit(lambda fz : brentq(fz, a=1e-12, b=1e8), static_argnums=(0))
# jit_brentq_l = jit(partial(brentq, a=-1e8, b=-1e-12), static_argnums=(0))
# jit_brentq_r = jit(partial(brentq, a=1e-12, b=1e8), static_argnums=(0))
# jit_brentq_l = jit(lambda f, args : brentq(f, args=args, a=-1e8, b=-1e-12), static_argnums=(0,1))
# jit_brentq_r = jit(lambda f, args : brentq(f, args=args, a=1e-12, b=1e8), static_argnums=(0,1))
# forward pass
# brentq(fz, args=(x, ds[0], theta, us[0,1]), a=-1e8, b=1e-12)

# brentq_partial_l = partial(brentq, a=-1e8, b=-1e-12)
# brentq_partial_r = partial(brentq, a=1e-12, b=1e8)
# @partial(jit, static_argnums=(0,3))
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
        # import ipdb; ipdb.set_trace()
        # z_L = jit_brentq_l(fz)
        # z_R = jit_brentq_r(fz)
        z_L = brentq(f_alpha, args=(x, d, theta, u1), a=-1e5, b=-1e-12)
        z_R = brentq(f_alpha, args=(x, d, theta, u1), a=1e-12, b=1e5)
        x_L = x + d*z_L
        x_R = x + d*z_R
        x = (1 - u2) * x_L + u2 * x_R

        xs.append(x)
        xLs.append(x_L)
        xRs.append(x_R)
        alphas.append(np.array([z_L,z_R]))

    return xs, xLs, xRs, alphas

# function for backwards pass
# @partial(jit, static_argnums=(0,8,9,10,11))
def backwards(S, theta, us, ds, xs, xLs, xRs, alphas,
              grad_theta, grad_x, grad_x_ad, dL_dx):

    D = xs[0].shape[0]
    dL_dtheta = np.zeros_like(theta)
    for s in range(S-1, -1, -1):

        u1 = us[s,0]
        u2 = us[s,1]
        z_L = alphas[s][0]
        z_R = alphas[s][1]

        # compute loss for current sample
        dL_dx_s = dL_dx(xs[s+1]) / S

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

    return dL_dtheta

# sample random variables
us = npr.rand(S,2)
ds = npr.randn(S, D)
ds = np.array([d / np.linalg.norm(d) for d in ds])
x = 0.1 * npr.randn(D) # initial x 


# loss function
xstar = npr.randn(D)
loss_fun = lambda x : np.mean((x - xstar)**2)

# gradient of loss with respect to x
dL_dx = grad(loss_fun)

# run forward pass
xs, xLs, xRs, alphas = forwards(S, theta, x, f_alpha, us, ds)
# xs, xLs, xRs, alphas = forwards(S, theta, x, f_alpha, us, ds, fz)
#

# jit_forwards = jit(forwards, static_argnums=(3))
# xs, xLs, xRs, alphas = jit_forwards(S, theta, x, f_alpha, us, ds, jit_brentq_l, jit_brentq_r)
# jit_forwards = jit(forwards, static_argnums=(0,3,6,7))
# xs, xLs, xRs, alphas = jit_forwards(S, theta, x, f_alpha, us, ds, fz)
# xs, xLs, xRs, alphas = jit_forwards(S, theta, x, f_alpha, us, ds)

# run backward pass
dL_dtheta = backwards(S, theta, us, ds, xs, xLs, xRs, alphas, grad_theta, grad_x, grad_x_ad, dL_dx)
print(dL_dtheta)
# compute gradient via finite differences
# dx = 1e-3
# dthetas = np.zeros(M)
# for m, v in enumerate(np.eye(M)):

#     theta1 = theta - dx * v
#     theta2 = theta + dx * v
#     xs1, xLs1, xRs1, alphas1 = forwards(S, theta1, x, f_alpha, us, ds)
#     xs2, xLs2, xRs2, alphas2 = forwards(S, theta2, x, f_alpha, us, ds)
#     loss1 = loss_fun(np.array(xs1[1:]))
#     loss2 = loss_fun(np.array(xs2[1:]))
#     dthetas = dthetas + (loss2 - loss1) / (2.0 * dx) * v

# print("Implicit: ", dL_dtheta)
# print("Numerical: ", dthetas)
# print("MSE: ", np.mean((dL_dtheta - dthetas)**2)) 