import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
import sys

from jax.lax import custom_root
from scipy.optimize import root, root_scalar

import matplotlib.pyplot as plt

D = 2
theta = np.zeros(D)
x = 0.5 * np.ones(D)
Sigma = np.eye(D)
xstar = np.array([1.1,0.5])

S = 4

# Sigma = np.array([[1.0, 0.5],[0.5,1.0]])
# Sigma = np.array([[1.0]])
def log_pdf(x, theta):
    d = x.shape[0]
    mu = theta
    normalizer = np.power(2*np.pi, d/2) * np.sqrt(np.linalg.det(Sigma))
    return -0.5 * (x - mu).T @ np.linalg.inv(Sigma) @ (x - mu) - np.log(normalizer)

def log_pdf_theta(theta, x):
    return log_pdf(x, theta)
def log_pdf_x(x, theta):
    return log_pdf(x, theta)
def log_pdf_ad(x, theta, a, d):
    return log_pdf(x + a * d, theta)

grad_x = grad(log_pdf_x)
grad_theta = grad(log_pdf_theta)
grad_x_ad = grad(log_pdf_ad)

# forward pass
us = npr.rand(S, 2)
ds = npr.randn(S, D)
ds = np.array([d / np.linalg.norm(d) for d in ds])

def forwards(theta, x, log_pdf, us, ds):

    S = us.shape[0]
    xs = [x]
    xLs = []
    xRs = []
    alphas = []
    for s in range(S):

        # if np.mod(s, 50) == 0:
        #     print(s)

        u1 = us[s,0]
        u2 = us[s,1]

        y = log_pdf(x, theta) + np.log(u1)

        # random direction
        d = ds[s]

        fz = lambda alpha : log_pdf(x + alpha * d, theta) - y
        res_L = root_scalar(fz, x0=-1e-4, method="brentq", bracket=[-1e8,-1e-12], xtol=1e-10)
        res_R = root_scalar(fz, x0=1e-4, method="brentq", bracket=[1e-12,1e8], xtol=1e-10)
        z_L = res_L.root
        z_R = res_R.root
        x_L = x + d*z_L
        x_R = x + d*z_R
        x = (1 - u2) * x_L + u2 * x_R
        # x = x + ( (1 - u2) * z_L + u2 * z_R ) * d
        xs.append(x)
        xLs.append(x_L)
        xRs.append(x_R)
        alphas.append(np.array([z_L,z_R]))

    return xs, xLs, xRs, alphas

xs, xLs, xRs, alphas = forwards(theta, x, log_pdf, us, ds)

loss_fun = lambda x : np.mean((x - xstar)**2)
dL_dx = grad(loss_fun)

dL_dtheta = np.zeros_like(theta)

# backwards pass
for s in range(S-1, -1, -1):

    print(s)
    u1 = us[s,0]
    u2 = us[s,1]
    z_L = alphas[s][0]
    z_R = alphas[s][1]

    dL_dx_s = dL_dx(xs[s+1]) / S
    if s < S-1:
        dL_dx_s += prev_dL_dx_s @ J_xs

    # 
    # L_grad_theta = -1.0 * (grad_theta(theta, xLs[s]) - grad_theta(theta, xs[s])) / np.dot(ds[s], grad_x(xLs[s], theta))
    # R_grad_theta = -1.0 * ( grad_theta(theta, xRs[s]) - grad_theta(theta, xs[s]) ) / np.dot(ds[s], grad_x(xRs[s], theta))
    L_grad_theta = -1.0 * (grad_theta(theta, xLs[s]) - grad_theta(theta, xs[s])) / np.dot(ds[s], grad_x_ad(xs[s], theta, z_L, ds[s]))
    R_grad_theta = -1.0 * (grad_theta(theta, xRs[s]) - grad_theta(theta, xs[s])) / np.dot(ds[s], grad_x_ad(xs[s], theta, z_R, ds[s]))

    #
    dL_dtheta_s = u2 * np.dot(dL_dx_s, ds[s]) * R_grad_theta + (1-u2) * np.dot(dL_dx_s, ds[s]) * L_grad_theta
    dL_dtheta += dL_dtheta_s

    # compute second term
    # if s < S-1:
    #     intr = prev_dL_dx_s @ J_xs
    #     second_term = u2 * np.dot(intr, ds[s]) * R_grad_theta + (1-u2) * np.dot(intr, ds[s]) * L_grad_theta
    #     dL_dtheta += second_term 

    # compute Jacobian of dx_s  / dx_{s-1}
    # L_grad_x = -1.0 * ( grad_x(xLs[s], theta) - grad_x(xs[s], theta) ) / np.dot(ds[s], grad_x(xLs[s], theta))
    # R_grad_x = -1.0 * ( grad_x(xRs[s], theta) - grad_x(xs[s], theta) ) / np.dot(ds[s], grad_x(xRs[s], theta))
    L_grad_x = -1.0 * ( grad_x_ad(xs[s], theta, z_L, ds[s]) - grad_x(xs[s], theta) ) / np.dot(ds[s], grad_x_ad(xs[s], theta, z_L, ds[s]))
    R_grad_x = -1.0 * ( grad_x_ad(xs[s], theta, z_R, ds[s]) - grad_x(xs[s], theta) ) / np.dot(ds[s], grad_x_ad(xs[s], theta, z_R, ds[s]))
    J_xs = np.eye(D) + u2 * np.outer(ds[s], R_grad_x) + (1-u2) * np.outer(ds[s], L_grad_x)

    prev_dL_dx_s = np.copy(dL_dx_s)

# ARE THERE SIGN ISSUES?
print("Implicit: ", dL_dtheta)

dx = 0.001
dthetas = np.zeros(D)
for d in range(D):

    theta1 = np.copy(theta)
    theta2 = np.copy(theta)
    theta1[d] -= dx 
    theta2[d] += dx
    xs1, xLs1, xRs1, alphas1 = forwards(theta1, x, log_pdf, us, ds)
    xs2, xLs2, xRs2, alphas2 = forwards(theta2, x, log_pdf, us, ds)
    loss1 = loss_fun(xs1[1:])
    loss2 = loss_fun(xs2[1:])
    dthetas[d] = (loss2 - loss1) / (2.0 * dx)

print("Numerical: ", dthetas)


# dx = 0.001
# theta1 = theta[0]-dx
# theta2 = theta[0]+dx
# xs1, xLs1, xRs1 = forwards(theta1, x, log_pdf, us, ds)
# xs2, xLs2, xRs2 = forwards(theta2, x, log_pdf, us, ds)
# loss1 = loss_fun(xs1[1:])
# loss2 = loss_fun(xs2[1:])
# dtheta1 = (loss2 - loss1) / (2.0 * dx)
# print("Numerical: ", dtheta1)
