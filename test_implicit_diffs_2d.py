# import jax.numpy as np
# import jax.random as random
# from jax import grad, jit, vmap
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
import sys
# import numpy.random as npr

from jax.lax import custom_root
from scipy.optimize import root, root_scalar

import matplotlib.pyplot as plt

def pdf(mu, Sigma, x1, x2):
    d = 2
    x = np.array([x1, x2])
    normalizer = np.power(2*np.pi, d/2) * np.sqrt(np.linalg.det(Sigma))
    return np.exp(-0.5 * (x - mu).T @ np.linalg.inv(Sigma) @ (x - mu)) / normalizer

def gaussian_pdf(x, mu, sig2):
    return 1.0 / np.sqrt(2.0 * np.pi* sig2) * np.exp(-0.5 * (x - mu) **2 / sig2)

def gaussian_log_pdf(x, mu, sig2):
    return -0.5 * (x - mu) **2 / sig2 - 0.5 * np.log(2.0 * np.pi * sig2)

def sqrt_term(y, sig2):
    return np.sqrt(-2.0 * sig2 * np.log(y * np.sqrt(2.0 * np.pi* sig2)))

def slice_sample(mu, Sigma, us, x):
    """
    x0 - initial x
    mu - mean
    us - uniform samples: size num_samples by 2
    sig2 - variance
    num_samples - number of samples
    """

    xs = []
    x1 = x[0]
    x2 = x[1]
    num_samples = us.shape[0]
    mu1 = mu[0]
    mu2 = mu[1]
    sig1 = Sigma[0,0]
    sig2 = Sigma[1,1]
    rho = Sigma[0,1]

    for i in range(num_samples):

        # x1
        u1 = us[i,0]
        u2 = us[i,1]
        mux = mu1 + sig1 / sig2 * rho * (x2 - mu2)
        sigx = (1-rho**2) * sig1
        y = gaussian_pdf(x1, mux, sigx) * u1
        L = mux - sqrt_term(y, sigx)
        R = mux + sqrt_term(y, sigx)
        x1 = (1-u2) * L + u2 * R

        # x1
        u3 = us[i,2]
        u4 = us[i,3]
        mux = mu2 + sig2 / sig1 * rho * (x1 - mu1)
        sigx = (1-rho**2) * sig2
        y = gaussian_pdf(x2, mux, sigx) * u3
        L = mux - sqrt_term(y, sigx)
        R = mux + sqrt_term(y, sigx)
        x2 = (1-u4) * L + u4 * R

        xs.append(np.array([x1,x2]))

    return np.array(xs)

def grad_theta(theta, Sigma, df, us, x):
    """
    df - function that returns gradient of loss function w.r.t x : df / dx
    """
    xs = []
    x1 = x[0]
    x2 = x[1]
    grad_thetas = []
    dxdthetas1 = []
    dxdthetas2 = []
    num_samples = us.shape[0]
    grad_theta = 0.0
    dxi_dx1 = 1.0
    dx2dx1s = []
    for i in range(num_samples):

        # x1
        u1 = us[i,0]
        u2 = us[i,1]
        y = u1 * pdf(theta, Sigma, x1, x2)
        fx = lambda x : pdf(theta, Sigma, x, x2) - y
        res_L = root_scalar(fx, x0=x1-1e-4, method="brentq", bracket=[-1e8,x1-1e-7])
        res_R = root_scalar(fx, x0=x1+1e-4, method="brentq", bracket=[x1+1e-7,1e8])
        x_L = res_L.root
        x_R = res_R.root
        f_theta_L = lambda theta : pdf(theta, Sigma, x_L, x2) - pdf(theta, Sigma, x1, x2) * u1 
        f_theta_R = lambda theta : pdf(theta, Sigma, x_R, x2) - pdf(theta, Sigma, x1, x2) * u1
        f_x = lambda x : pdf(theta, Sigma, x, x2)
        g_L = grad(f_theta_L)
        g_R = grad(f_theta_R)
        g_x = grad(f_x)
        dxL_dtheta = - 1.0 / g_x(x_L) * g_L(theta)
        dxR_dtheta = - 1.0 / g_x(x_R) * g_R(theta)
        dxdtheta1 = (1.0 - u2) * dxL_dtheta + u2 * dxR_dtheta

        if i > 0:

            # dependence of x1 on previous x2
            g_x2L = grad(lambda x: pdf(theta, Sigma, x_L, x) - pdf(theta, Sigma, x1, x) * u1)
            g_x2R = grad(lambda x: pdf(theta, Sigma, x_R, x) - pdf(theta, Sigma, x1, x) * u1)
            dx1dx2 = -(1 - u2) * 1.0 / g_x(x_L) * g_x2L(x2) - u2 * 1.0 / g_x(x_R) * g_x2R(x2)
            dxdtheta1 = dxdtheta1 + dx1dx2 * dxdthetas2[-1]


            # dependence of x1 on previous x1
            dxLdx1 = g_x(x1) * u1 / g_x(x_L)
            dxRdx1 = g_x(x1) * u1 / g_x(x_R)
            dx2dx1 = (1-u2) * dxLdx1 + u2 * dxRdx1
            dxdtheta1 = dxdtheta1 + dx2dx1 * dxdthetas1[-1]

        dxdthetas1.append(dxdtheta1)
        x1 = (1.0 - u2) * x_L + u2 * x_R

        # x2
        u1 = us[i,2]
        u2 = us[i,3]
        y = u1 * pdf(theta, Sigma, x1, x2)
        fx = lambda x : pdf(theta, Sigma, x1, x) - y
        res_L = root_scalar(fx, x0=x2-1e-4, method="brentq", bracket=[-1e8,x2-1e-7])
        res_R = root_scalar(fx, x0=x2+1e-4, method="brentq", bracket=[x2+1e-7,1e8])
        x_L = res_L.root
        x_R = res_R.root
        f_theta_L = lambda theta : pdf(theta, Sigma, x1, x_L) - pdf(theta, Sigma, x1, x2) * u1
        f_theta_R = lambda theta : pdf(theta, Sigma, x1, x_R) - pdf(theta, Sigma, x1, x2) * u1
        f_x = lambda x : pdf(theta, Sigma, x1, x)
        g_L = grad(f_theta_L)
        g_R = grad(f_theta_R)
        g_x = grad(f_x)
        dxL_dtheta = - 1.0 / g_x(x_L) * g_L(theta)
        dxR_dtheta = - 1.0 / g_x(x_R) * g_R(theta)
        g_x1L = grad(lambda x: pdf(theta, Sigma, x, x_L) - pdf(theta, Sigma, x, x2) * u1)
        g_x1R = grad(lambda x: pdf(theta, Sigma, x, x_R) - pdf(theta, Sigma, x, x2) * u1)
        dx2dx1 = -(1 - u2) * 1.0 / g_x(x_L) * g_x1L(x1) - u2 * 1.0 / g_x(x_R) * g_x1R(x1)

        dxdtheta2 = (1.0 - u2) * dxL_dtheta + u2 * dxR_dtheta
        dxdtheta2 = dxdtheta2 + dx2dx1 * dxdtheta1
        dxLdx1 = g_x(x2) * u1 / g_x(x_L)
        dxRdx1 = g_x(x2) * u1 / g_x(x_R)
        dx2dx1 = (1-u2) * dxLdx1 + u2 * dxRdx1
        if i > 0:
            dxdtheta2 = dxdtheta2 + dx2dx1 * dxdthetas2[-1]

        dxdthetas2.append(dxdtheta2)
        x2 = (1.0 - u2) * x_L + u2 * x_R

        xs.append(np.array([x1,x2]))
        # import ipdb; ipdb.set_trace()
        grad_theta += df(np.array([x1,x2])) @ np.array([dxdtheta1,dxdtheta2])
  

    grad_theta /= num_samples

    return np.array(xs), grad_theta, dxdthetas1, dxdthetas2

# initialize sig2, xstar
# sig2 = 2.0
xstar = np.array([1.1,0.5])
x0 = np.array([0.1,0.1])
mu = np.array([0.0,0.0])
# Sigma = np.eye(2)
Sigma = np.array([[1.0,0.5],[0.5,1.0]])
# sample random variables
if len(sys.argv) > 1:
    num_samples = int(sys.argv[1])
else:
    num_samples = 1
print("Number of samples: ", num_samples)
us = npr.rand(num_samples,4)

# loss for Gaussian slice sampling
def loss(mu, x, us):
    xs = slice_sample(mu, Sigma, us, x)
    return np.mean((xs-xstar)**2) 
g_jax = grad(loss)

# def loss_reparam(mu, x, us):
#     xs = mu + npr.randn(2)
#     # xs = slice_sample(mu, Sigma, us, x)
#     return np.mean((xs-xstar)**2)
# g_jax = grad(loss_reparam)

# compute loss 
g1 = g_jax(mu, x0, us)

# compute loss via root finding slice sampling
loss_fun = lambda x : np.mean((x - xstar)**2)
df = grad(loss_fun)
xs, grad_theta_jax, dxdthetas1, dxdthetas2 = grad_theta(mu, Sigma, df, us, x0)

print("Jax: ", g1)
print("Root: ", grad_theta_jax)

# test finite differences
dx = 0.001
mu1 = np.array([mu[0]-dx,mu[1]])
mu2 = np.array([mu[0]+dx,mu[1]])
xs1, _, _, _ = grad_theta(mu1, Sigma, df, us, x0)
xs2, _, _, _ = grad_theta(mu2, Sigma, df, us, x0)
loss1 = loss_fun(xs1)
loss2 = loss_fun(xs2)
dmu1 = (loss2 - loss1) / (2.0 * dx)
mu1 = np.array([mu[0],mu[1]-dx])
mu2 = np.array([mu[0],mu[1]+dx])
xs1, _, _, _ = grad_theta(mu1, Sigma, df, us, x0)
xs2, _, _, _ = grad_theta(mu2, Sigma, df, us, x0)
loss1 = loss_fun(xs1)
loss2 = loss_fun(xs2)
dmu2 = (loss2 - loss1) / (2.0 * dx)
print("Finite differences: ", np.array([dmu1, dmu2]))
# num_sampless = [1,2,5]
# thetas_jax_all = []
# thetas_root_all = []
# losses_all = []
# for num_samples in num_sampless:
#     ## Uncomment if you want to run experiment
#     theta_jax = np.array([-0.1,-0.1])
#     theta_root = np.array([-0.1,-0.1])
#     thetas_jax = []
#     thetas_jax.append(theta_jax)
#     thetas_root = []
#     thetas_root.append(theta_root)
#     x = np.array([0.05,0.05])
#     xs = [x]
#     alpha = 0.01 # learning rate
#     losses_root = []
#     num_iters=500
#     # num_samples=1
#     a0 = 0.3
#     gam = 0.2
#     # run many times
#     for i in range(num_iters):
#         print(i)

#         # draw samples
#         us = npr.rand(num_samples,4)

#         # learning rate
#         alpha_t = a0 / (1 + gam * (i+1))

#         # jax
#         # print(xs[-1])
#         dfdtheta_jax = g_jax(theta_jax, xs[-1], us)
#         theta_jax = theta_jax - dfdtheta_jax * alpha_t
#         thetas_jax.append(theta_jax)
#         # losses_jax.append(np.mean(loss(x_jax)))

#         # root
#         xs, dfdtheta_root, _, _ = grad_theta(theta_root, Sigma, df, us, xs[-1])
#         theta_root = theta_root - dfdtheta_root * alpha_t
#         thetas_root.append(theta_root)
#         losses_root.append(np.mean(loss2(np.array(xs))))

#     thetas_jax_plot = np.array(thetas_jax)
#     thetas_root_plot = np.array(thetas_root)
#     thetas_jax_all += [thetas_jax_plot]
#     thetas_root_all += [thetas_root_plot]
#     losses_all += [np.array(losses_root)]

# jax_colors = [[1.0,0.7,0.7], [1.0,0.35,0.35], [1.0,0.0,0.0]]
# root_colors = [[0.7,0.7,1.0], [0.35,0.35,1.0], [0.0,0.0,1.0]]
# # jax_colors = [[1.0,0.75,0.75], [1.0,0.5,0.5], [1.0,0.25,0.25], [1.0,0.0,0.0]]
# # root_colors = [[0.75, 0.75, 1.0], [0.5,0.5,1.0], [0.25,0.25,1.0], [0.0,0.0,1.0]]
# plt.ion()
# plt.figure()
# plt.subplot(311)
# plt.plot([0,num_iters],[xstar[0], xstar[0]],'k--',linewidth=1)
# for i, num_samples in enumerate(num_sampless):
#     plt.plot(thetas_jax_all[i][:,0],'-', color=jax_colors[i],label="autodiff, " + str(num_samples))
#     plt.plot(thetas_root_all[i][:,0],'--', color=root_colors[i],label="implicit, " + str(num_samples))
# plt.legend()
# # plt.plot(thetas_root,'b',label="root")
# plt.ylabel("$\mu_1$")
# plt.subplot(312)
# plt.plot([0,num_iters],[xstar[1], xstar[1]],'k--',linewidth=1)
# for i, num_samples in enumerate(num_sampless):
#     plt.plot(thetas_jax_all[i][:,1],'-', color=jax_colors[i],label="root")
#     plt.plot(thetas_root_all[i][:,1],'--', color=root_colors[i],label="jax")
# # plt.plot(thetas_root,'b',label="root")
# plt.ylabel("$\mu_2$")
# plt.subplot(313)
# for i, num_samples in enumerate(num_sampless):
#     curr_loss = losses_all[i]
#     smoothed_loss = [np.mean(curr_loss[max(i-1,0):i+2]) for i in range(len(curr_loss))]
#     plt.plot(smoothed_loss,'-', color=root_colors[i],label="jax")
# plt.xlabel("iteration")
# plt.ylabel("loss")
# plt.tight_layout()
