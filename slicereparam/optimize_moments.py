import autograd.numpy as np 
from autograd import grad
from autograd.misc import flatten
import numpy.random as npr
# npr.seed(0)

from scipy.optimize import root_scalar, brentq
import matplotlib.pyplot as plt
from tqdm.auto import trange
import sys

# Set up params
D = 3   # number of dimensions
S = 5  # number of samples
M = 2*D   # number of parameters

# parameters are mean and diagonal log covariance
# _theta = [0.0 * np.ones(D), 0.0 * np.ones(D)]
_theta = [0.1 * npr.rand(D), 0.1 * npr.rand(D)]
def _log_pdf(x, params):
    mu = params[0]
    sigma_diag = np.exp(params[1])
    return np.sum(-0.5 * (x - mu) **2 / sigma_diag)# - 0.5 *  np.log(np.linalg.det(np.diag(sigma_diag)))

def _log_pdf(x, params):
    mu = params[0]
    sigma_diag = np.exp(params[1])
    return np.sum(-0.5 * (x - mu) **2 / sigma_diag)# - 0.5 *  np.log(np.linalg.det(np.diag(sigma_diag)))

def _log_pdf2(x, params):
    mu = params[0]
    Sigma = np.diag(np.exp(params[1]))
    return -0.5 * (x - mu).T @ np.linalg.inv(Sigma) @ (x - mu) #- 0.5 *  np.log(np.linalg.det(Sigma))


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
ys = theta[:D] + np.sqrt(np.exp(theta[D:])) * npr.randn(S, D)

# loss function
xstar = npr.randn(D)
# def _loss(x, y, params):
#     # x_mean = np.mean(x, axis=0)
#     # x2_mean = np.mean(x**2, axis=0)
#     # y_mean = np.mean(y, axis=0)
#     # y2_mean = np.mean(y**2, axis=0)
#     y_mean = np.mean(y, axis=0)
#     y2_mean = np.mean(y**2, axis=0)
#     return (np.sum((x-y_mean)**2) + np.sum((x**2-y2_mean)**2))*S

def features(xs):
    # D = xs.shape[1]
    S = xs.shape[0]
    x_mean = np.mean(xs,axis=0)
    x_demeaned = xs - x_mean
    x2_mean = np.mean(x_demeaned**2, axis=0) * S / (S-1)
    # x1x2 = np.mean(x_demeaned[:,0] * x_demeaned[:,1]) * S / (S-1)
 
    return np.hstack((x_mean, x2_mean))#, x1x2))
    # return x_mean
    # return np.hstack((x_mean, x2_mean))
    # return np.hstack((x_mean, x_cov))

def _total_loss(xs, ys, params):
    # x_mean = np.mean(xs, axis=0)
    # x2_mean = np.mean(xs**2, axis=0)
    # y_mean = np.mean(ys, axis=0)
    # y2_mean = np.mean(ys**2, axis=0)
    # return np.sum((x_mean-y_mean)**2) + 1.0 * np.sum((x2_mean-y2_mean)**2)
    phi_x = features(xs)
    phi_y = features(ys)
    return np.sqrt(np.sum((phi_x - phi_y)**2))

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
print("Explicit: ", grad_loss_reparam(theta, ds, ys))
# ys_new = theta[:D] + np.sqrt(np.exp(theta[D:])) * npr.randn(1000000, D)
# print("Explicit: ", grad_loss_reparam(theta, npr.randn(1000000, 2), ys_new))
# print("Explicit: ", grad_loss_reparam(theta, npr.randn(1000000,2), ys))
# exact_grad = np.concatenate(( 2.0 * (theta[:D] - xstar), np.exp(theta[D:]) - 0.5 )) 
# print("Exact   : ", exact_grad)

# compute gradient via finite differences
# dx = 1e-3
# M = theta.shape[0]
# dthetas = np.zeros_like(theta)
# for m, v in enumerate(np.eye(M)):

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
theta = np.hstack([0.5 * npr.randn(D), np.log(1.0 + 0.5 * npr.randn(D))])
# theta = 0.1 * npr.randn(M)
theta_reparam = np.copy(theta)
thetas = [theta]
thetas_reparam = [theta]
xs = [x]
# losses = [loss(x, theta)]
losses = [0.0]
if len(sys.argv) > 1:
    S = int(sys.argv[1])
else:
    S = 1
num_iters=500

# learning rate params
a0 = 0.2
gam = 0.02

# plt.figure()
# plt.plot(np.arange(num_iters), a0 / (1 + gam * (np.arange(num_iters)+1)))

# TRUE VARIANCE
# true_var = 1.0
true_var = np.exp(0.2 * npr.randn(D))
pbar = trange(num_iters)
pbar.set_description("Loss: {:.1f}".format(losses[0]))
# ys = xstar + np.sqrt(true_var) * npr.randn(100000, D)
print("Features: ", features(ys))
plt.figure()
for i in range(num_iters):

    us = npr.rand(S, 2)
    ds = npr.randn(S, D)
    norm_ds = np.array([d / np.linalg.norm(d) for d in ds])
    ys = xstar + np.sqrt(true_var) * npr.randn(S, D)

    # forward pass
    # x0 = xs[-1]
    x0 = theta[:D] + np.sqrt(np.exp(theta[D:])) * npr.randn(D)
    xs, xLs, xRs, alphas = forwards(S, theta, x0, f_alpha, us, norm_ds)

    # backwards pass
    dL_dxs = loss_grad_xs(xs[1:], ys, theta)
    dL_dtheta = backwards(S, theta, us, norm_ds, xs, xLs, xRs, alphas,
                      grad_theta, grad_x, grad_x_ad, dL_dxs, loss_grad_params, ys, bS=0)
    #dL_dtheta += loss_grad_params(theta, xs[1:])
    # update parameters
    alpha_t = a0 / (1 + gam * (i+1)) # learning rate 
    theta = theta - dL_dtheta * alpha_t
    thetas.append(theta)
    losses.append(loss(xs[1:], ys, theta))

    # reparameterization trick
    dL_dtheta_reparam = grad_loss_reparam(theta_reparam, npr.randn(S,D), ys)
    theta_reparam = theta_reparam - dL_dtheta_reparam * alpha_t
    thetas_reparam.append(theta_reparam)

    # if np.mod(i,25)==0:
        # plt.plot(xs[:,0], xs[:,1])
        # plt.pause(0.1)

    pbar.set_description("Loss: {:.1f}".format(losses[-1]))
    pbar.update()

pbar.close()

thetas_plot = np.array(thetas)
thetas_reparam_plot = np.array(thetas_reparam)

plt.ion()
plt.figure()
plt.subplot(211)
for d in range(D):
    plt.plot([0,num_iters],xstar[d]*np.ones(2),'k--')
plt.plot(thetas_plot[:,:D],'b')
plt.plot(thetas_reparam_plot[:,:D],'r--')
plt.ylabel("$\mu$")
plt.title("# Samples: " + str(S))
plt.subplot(212)
for d in range(D):
    plt.plot([0,num_iters],true_var[d]*np.ones(2),'k--', label="true" if d == 0 else None)
    plt.plot(np.exp(thetas_plot[:,D+d]),'b', label="slice" if d == 0 else None)
    plt.plot(np.exp(thetas_reparam_plot[:,D+d]),'r--', label="reparam" if d == 0 else None)
# plt.plot(np.exp(thetas_plot[:,2]),'b', label="slice")
# plt.plot(np.exp(thetas_plot[:,3]),'b')
# plt.plot(np.exp(thetas_reparam_plot[:,2]),'r--', label="reparam")
# plt.plot(np.exp(thetas_reparam_plot[:,3]),'r--')
plt.xlabel("iteration")
plt.ylabel("$\sigma^2$")
plt.ylim([0.0,2.0])
plt.legend()
plt.tight_layout()
# plt.savefig("optimize_moments_dim" + str(D) + "_samples" + str(S) + ".png")