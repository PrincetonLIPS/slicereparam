import jax.numpy as np
import jax.random as random
from jax import grad, jit, vmap

import numpy.random as npr

from jax.lax import custom_root
from scipy.optimize import root, root_scalar

import matplotlib.pyplot as plt

# pdf function
# def pdf(x, mu, sig2):
    # return 1.0 / np.sqrt(2.0 * np.pi* sig2) * np.exp(-0.5 * (x - mu) **2 / sig2)

def pdf(x, mu, sig2):
    return 1.0 / (2.0 * sig2) * np.exp(- np.abs(x - mu) / sig2)


# slice sampling gradient

# mu = 0.1
sig2 = 1.0
# x0 = 0.05
# u1 = npr.rand()
# u2 = npr.rand()
# y = u1 * pdf(x0, mu, sig2)

def grad_theta(theta, df, x0, u1, u2):
    """
    df - function that returns gradient of loss function w.r.t x : df / dx
    """

    y = u1 * pdf(x0, theta, sig2)
    fx = lambda x : pdf(x, theta, sig2) - y

    # find roots using scipy.optimize.root_scalar
    res_L = root_scalar(fx, x0=x0-1e-3, method="brentq", bracket=[-1e4,x0-1e-3])
    res_R = root_scalar(fx, x0=x0+1e-3, method="brentq", bracket=[x0+1e-3,1e4])
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

    x = (1 - u2) * x_L + u2 * x_R
    grad_theta = df(x) * dxdtheta

    return x, grad_theta

xstar = 5.0
loss = lambda x : (x - xstar)**2
df = grad(loss)

# u1 = npr.rand()
# u2 = npr.rand()
# grad_theta(0.0, df, 0.05, u1, u2)

theta = -1.0
thetas = []
thetas.append(theta)
x = -0.95
alpha = 0.02 # learning rate
losses = []
losses.append(loss(x))
num_iters=250
# run many times
for i in range(num_iters):
    print(i)
    u1 = npr.rand()
    u2 = npr.rand()
    x, dfdtheta = grad_theta(theta, df, x, u1, u2)
    theta -= dfdtheta * alpha
    thetas.append(theta)
    losses.append(loss(x))

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
