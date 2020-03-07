# import jax.numpy as np
# import jax.random as random
# from jax import grad, jit, vmap
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

# import numpy.random as npr

from jax.lax import custom_root
from scipy.optimize import root, root_scalar

import matplotlib.pyplot as plt

# pdf function
# def pdf(x, mu, sig2):
#     return 1.0 / np.sqrt(2.0 * np.pi* sig2) * np.exp(-0.5 * (x - mu) **2 / sig2)

# Laplace
def pdf(x, mu, sig2):
    return 1.0 / (2.0 * sig2) * np.exp(- np.abs(x - mu) / sig2)

# Log normal
# def pdf(x, mu, sig2):
#     mu = np.exp(mu)
#     if x > 1e-6:
#         out = 1.0 / (x * np.sqrt(2.0 * np.pi* sig2)) * np.exp(-0.5 * (np.log(x) - mu) **2 / sig2)
#     else:
#         out = 0.0
#     return out



# slice sampling gradient

# mu = 0.1
sig2 = 1.0
# x0 = 0.05
# u1 = npr.rand()
# u2 = npr.rand()
# y = u1 * pdf(x0, mu, sig2)

def grad_theta(theta, df, x0, us):
    """
    df - function that returns gradient of loss function w.r.t x : df / dx
    """
    xs = np.array([])
    grad_thetas = []
    dxdthetas = []
    num_samples = us.shape[0]
    grad_theta = 0.0
    for i in range(num_samples):

        u1 = us[i,0]
        u2 = us[i,1]
        y = u1 * pdf(x0, theta, sig2)
        fx = lambda x : pdf(x, theta, sig2) - y

        # find roots using scipy.optimize.root_scalar
        res_L = root_scalar(fx, x0=x0-1e-3, method="brentq", bracket=[-1e9,x0-1e-9])
        res_R = root_scalar(fx, x0=x0+1e-3, method="brentq", bracket=[x0+1e-9,1e9])
        x_L = res_L.root
        x_R = res_R.root

        f_theta_L = lambda theta : pdf(x_L, theta, sig2) - pdf(x0, theta, sig2) * u1
        f_theta_R = lambda theta : pdf(x_R, theta, sig2) - pdf(x0, theta, sig2) * u1
        f_x = lambda x : pdf(x, theta, sig2)

        g_L = grad(f_theta_L)
        g_R = grad(f_theta_R)
        g_x = grad(f_x)

        dxL_dtheta = - 1.0 / g_x(x_L) * g_L(theta)
        dxR_dtheta = - 1.0 / g_x(x_R) * g_R(theta)
        dxdtheta = (1.0 - u2) * dxL_dtheta + u2 * dxR_dtheta
        if i > 0:
            dxLdx1 = g_x(x0) * u1 / g_x(x_L)
            dxRdx1 = g_x(x0) * u1 / g_x(x_R)
            dx2dx1 = (1-u2) * dxLdx1 + u2 * dxRdx1
            dxdtheta += dx2dx1 * dxdthetas[-1]
        dxdthetas.append(dxdtheta)

        x0 = (1.0 - u2) * x_L + u2 * x_R
        xs = np.append(xs, x0)
        grad_theta += df(x0) * dxdtheta
        
    grad_theta /= num_samples

    return xs, grad_theta

xstar = 25.0
loss = lambda x : (x - xstar)**2
df = grad(loss)

# u1 = npr.rand()
# u2 = npr.rand()
# grad_theta(0.0, df, 0.05, u1, u2)

theta = np.log(3.0)
thetas = []
thetas.append(theta)
x = [0.95]
alpha = 0.01 # learning rate
losses = []
# losses.append(loss(x[-1]))
num_iters=500
num_samples=3
a0 = 0.2
gam = 0.5
# run many times
for i in range(num_iters):
    print(i)
    us = npr.rand(num_samples,2)
    # u1 = npr.rand()
    # u2 = npr.rand()
    x, dfdtheta = grad_theta(theta, df, x[-1], us)
    alpha_t = a0 / (1 + gam * (i+1))
    theta -= dfdtheta * alpha_t
    thetas.append(theta)
    losses.append(np.mean(loss(x)))

plt.ion()
plt.figure()
plt.subplot(211)
plt.title("min E[(x - 5)^2], x ~ Laplace($\mu$, 1)")
plt.plot(thetas,'b')
# plt.plot(mu2s,'r')
plt.ylabel("$\mu$")
plt.plot([0,num_iters],[xstar, xstar],'k--',linewidth=0.5)
plt.subplot(212)
plt.plot(losses,'b', label="Slice")
plt.xlabel("iteration")
plt.ylabel("loss")
# plt.plot(loss2,'r', label="Reparam")
# plt.legend()
plt.tight_layout()
