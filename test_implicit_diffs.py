import jax.numpy as np
import jax.random as random
from jax import grad, jit, vmap

import numpy.random as npr

from jax.lax import custom_root
from scipy.optimize import root, root_scalar

# pdf function
def gaussian_pdf(x, mu, sig2):
    return 1.0 / np.sqrt(2.0 * np.pi* sig2) * np.exp(-0.5 * (x - mu) **2 / sig2)

# slice sampling gradient
mu = 0.1
sig2 = 2.0
x0 = 0.05
u1 = npr.rand()
u2 = npr.rand()
y = u1 * gaussian_pdf(x0, mu, sig2)

fx = lambda x : gaussian_pdf(x, mu, sig2) - y

# find roots using scipy.optimize.root
# res_L = root(fx, x0-0.1)
# res_R = root(fx, x0+0.1)
# x_L = res_L.x[0]
# x_R = res_R.x[0]

# find roots using scipy.optimize.root_scalar
res_L = root_scalar(fx, x0=x0-1e-3, method="brentq", bracket=[x0-1e-3,-100])
res_R = root_scalar(fx, x0=x0+1e-3, method="brentq", bracket=[x0+1e-3,100])
x_L = res_L.root
x_R = res_R.root


f_theta_L = lambda mu : gaussian_pdf(x_L, mu, sig2) - gaussian_pdf(x0, mu, sig2) * u1
f_theta_R = lambda mu : gaussian_pdf(x_R, mu, sig2) - gaussian_pdf(x0, mu, sig2) * u1
f_x = lambda x : gaussian_pdf(x, mu, sig2)

g_L = grad(f_theta_L)
g_R = grad(f_theta_R)
g_x = grad(f_x)

dxL_dtheta = - 1.0 / g_x(x_L) * g_L(mu)
dxR_dtheta = - 1.0 / g_x(x_R) * g_R(mu)
dxdtheta = (1.0 - u2) * dxL_dtheta + u2 * dxR_dtheta

l_theta = lambda mu : mu - np.sqrt(-2.0 * sig2 * np.log(gaussian_pdf(x0, mu, sig2) * u1 * np.sqrt(2.0*np.pi*sig2)))
r_theta = lambda mu : mu + np.sqrt(-2.0 * sig2 * np.log(gaussian_pdf(x0, mu, sig2) * u1 * np.sqrt(2.0*np.pi*sig2)))
f_theta_analytic = lambda mu : (1.0 - u2) * l_theta(mu) + u2 * r_theta(mu)
g_theta_analytic = grad(f_theta_analytic)

print(dxdtheta)
print(g_theta_analytic(mu))


