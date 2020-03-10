import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
import sys

from jax.lax import custom_root
from scipy.optimize import root, root_scalar

import matplotlib.pyplot as plt

def gaussian_pdf(x, mu, sig2):
    return 1.0 / np.sqrt(2.0 * np.pi* sig2) * np.exp(-0.5 * (x - mu) **2 / sig2)

def gaussian_log_pdf(x, mu, sig2):
    return -0.5 * (x - mu) **2 / sig2 - 0.5 * np.log(2.0 * np.pi * sig2)

mu = 0.0
sig2 = 1.0
xmin = -4.0
xmax = 4.0
dx = 0.01
x_range = np.arange(xmin,xmax+dx,dx)
x0 = npr.randn()
# x0 = 0.05

# random
u1 = npr.rand()
u2 = npr.rand()
e1 = -np.log(u1)

# pdf 
y_pdf = gaussian_pdf(x0, mu, sig2) * u1 
fx = lambda x : gaussian_pdf(x, mu, sig2) - y_pdf
res_L = root_scalar(fx, x0=x0-1e-4, method="brentq", bracket=[-1e8,x0-1e-7])
res_R = root_scalar(fx, x0=x0+1e-4, method="brentq", bracket=[x0+1e-7,1e8])
xL_pdf = res_L.root
xR_pdf = res_R.root
slice_pdf = np.array([xL_pdf, xR_pdf])

# gradient pdf
f_theta_L = lambda theta : gaussian_pdf(xL_pdf, theta, sig2) - gaussian_pdf(x0, theta, sig2) * u1 
f_theta_R = lambda theta : gaussian_pdf(xR_pdf, theta, sig2) - gaussian_pdf(x0, theta, sig2) * u1 
f_x = lambda x : gaussian_pdf(x, mu, sig2)
g_L = grad(f_theta_L)
g_R = grad(f_theta_R)
g_x = grad(f_x)
dxL_dtheta = - 1.0 / g_x(xL_pdf) * g_L(mu)
dxR_dtheta = - 1.0 / g_x(xR_pdf) * g_R(mu)
dxdtheta_pdf = (1.0 - u2) * dxL_dtheta + u2 * dxR_dtheta

# log pdf
y_logpdf = gaussian_log_pdf(x0, mu, sig2) - e1
fx = lambda x : gaussian_log_pdf(x, mu, sig2) - y_logpdf
res_L = root_scalar(fx, x0=x0-1e-4, method="brentq", bracket=[-1e8,x0-1e-7])
res_R = root_scalar(fx, x0=x0+1e-4, method="brentq", bracket=[x0+1e-7,1e8])
xL_logpdf = res_L.root
xR_logpdf = res_R.root
slice_logpdf = np.array([xL_logpdf, xR_logpdf])

# gradient log pdf
f_theta_L = lambda theta : gaussian_log_pdf(xL_logpdf, theta, sig2) - gaussian_log_pdf(x0, theta, sig2) 
f_theta_R = lambda theta : gaussian_log_pdf(xR_logpdf, theta, sig2) - gaussian_log_pdf(x0, theta, sig2)  
f_x = lambda x : gaussian_log_pdf(x, mu, sig2)
g_L = grad(f_theta_L)
g_R = grad(f_theta_R)
g_x = grad(f_x)
dxL_dtheta = - 1.0 / g_x(xL_logpdf) * g_L(mu)
dxR_dtheta = - 1.0 / g_x(xR_logpdf) * g_R(mu)
dxdtheta_logpdf = (1.0 - u2) * dxL_dtheta + u2 * dxR_dtheta

print("Slice using    pdf: ", slice_pdf)
print("Slice using logpdf: ", slice_logpdf)
print("Gradient using    pdf: ", dxdtheta_pdf)
print("Gradient using logpdf: ", dxdtheta_logpdf)

plt.ion()
plt.figure()
plt.subplot(211)
plt.plot(x_range, gaussian_pdf(x_range, mu, sig2))
plt.plot(x0, y_pdf, 'k.')
plt.plot(slice_pdf, gaussian_pdf(slice_pdf, mu, sig2), 'k')
plt.axvline(slice_pdf[0], color='k', linestyle='--')
plt.axvline(slice_pdf[1], color='k', linestyle='--')
plt.xlim([xmin,xmax])
plt.subplot(212)
plt.plot(x_range, gaussian_log_pdf(x_range, mu, sig2))
plt.plot(slice_logpdf, gaussian_log_pdf(slice_logpdf, mu, sig2), 'k')
plt.plot(x0, y_logpdf, 'k.')
plt.axvline(slice_logpdf[0], color='k', linestyle='--')
plt.axvline(slice_logpdf[1], color='k', linestyle='--')
plt.xlim([xmin,xmax])
