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

def sqrt_term(y, sig2):
    return np.sqrt(-2.0 * sig2 * np.log(y * np.sqrt(2.0 * np.pi* sig2)))

# pdf function
def gaussian_pdf(x, mu, sig2):
    return 1.0 / np.sqrt(2.0 * np.pi* sig2) * np.exp(-0.5 * (x - mu) **2 / sig2)

def pdf(x, thetas):
    mu = thetas[0]
    # sig2 = np.exp(thetas[1])
    sig2 = thetas[1]
    return 1.0 / np.sqrt(2.0 * np.pi* sig2) * np.exp(-0.5 * (x - mu) **2 / sig2)

def slice_sample(params, us, x):
    """
    x0 - initial x
    mu - mean
    us - uniform samples: size num_samples by 2
    sig2 - variance
    num_samples - number of samples
    """

    mu = params[0]
    sig2 = params[1]
    xs = np.array([])
    num_samples = us.shape[0]

    for i in range(num_samples):

        # sample uniform random variables
        # u1 = random.uniform(key)
        # u2 = random.uniform(key)
        u1 = us[i,0]
        u2 = us[i,1]
        
        # sample height
        y = gaussian_pdf(x, mu, sig2) * u1

        # get boundaries
        L = mu - sqrt_term(y, sig2)
        R = mu + sqrt_term(y, sig2)

        # sample new location
        # x = L + (R-L) * u2
        x = (1-u2) * L + u2 * R
        xs = np.append(xs, x)

    return xs

def fx(x, y, thetas):
    return pdf(x, thetas) - y

def f_thetas_LR(thetas, x_LR, x0, u1):
    return pdf(x_LR, thetas) - pdf(x0, thetas) * u1

grad_fx = grad(fx)
grad_LR = grad(f_thetas_LR)
grad_x = grad(pdf)

def grad_theta(thetas, df, x0, us, dloss_theta):
    """
    df - function that returns gradient of loss function w.r.t x : df / dx
    """
    xs = []
    grad_thetas = []
    dxdthetas = []
    num_samples = us.shape[0]
    grad_theta = 0.0
    dxi_dx1 = 1.0
    dx2dx1s = []
    for i in range(num_samples):

        u1 = us[i,0]
        u2 = us[i,1]
        y = u1 * pdf(x0, thetas)

        # find roots using scipy.optimize.root_scalar
        res_L = root_scalar(lambda x : fx(x, y, thetas), 
                            x0=x0-1e-4, method="brentq", bracket=[-1e8,x0-1e-5])
        res_R = root_scalar(lambda x : fx(x, y, thetas), 
                            x0=x0+1e-4, method="brentq", bracket=[x0+1e-5,1e8])
        x_L = res_L.root
        x_R = res_R.root

        dxL_dtheta = - 1.0 / grad_x(x_L, thetas) * grad_LR(thetas, x_L, x0, u1)
        dxR_dtheta = - 1.0 / grad_x(x_R, thetas) * grad_LR(thetas, x_R, x0, u1)
        dxdtheta = (1.0 - u2) * dxL_dtheta + u2 * dxR_dtheta

        dxLdx1 = grad_x(x0, thetas) * u1 / grad_x(x_L, thetas)
        dxRdx1 = grad_x(x0, thetas) * u1 / grad_x(x_R, thetas)
        dx2dx1 = (1-u2) * dxLdx1 + u2 * dxRdx1
        dx2dx1s.append(dx2dx1)
        if i > 0:
            dxdtheta += dx2dx1 * dxdthetas[-1]

        dxdthetas.append(dxdtheta)

        x0 = (1.0 - u2) * x_L + u2 * x_R
        xs.append(x0)
        grad_theta += df(x0) * dxdtheta 
  

    grad_theta /= num_samples
    grad_theta += dloss_theta(thetas)

    return xs, grad_theta, grad_thetas, dxdthetas, dx2dx1s

# initialize sig2, xstar
sig2 = 2.0
xstar = 1.1
x0 = 0.05
mu1 = 0.1

# sample random variables
num_samples = 3
us = npr.rand(num_samples,2)

# loss for Gaussian slice sampling
def loss(params, x, us):
    x_new = slice_sample(params, us, x)
    loss = np.mean((x_new - xstar )**2) - np.log(np.sqrt(2.0 * np.pi * np.exp(1) * params[1]))
    return loss
g_jax = grad(loss)

# compute loss 
params = np.array([mu1, sig2])
g1 = g_jax(params, x0, us)

# compute loss via root finding slice sampling
loss_x = lambda x : (x - xstar)**2
loss_theta = lambda theta : - np.log(np.sqrt(2.0 * np.pi * np.exp(1) * theta[1]))
df = grad(loss_x)
dloss_theta = grad(loss_theta)
xs, grad_theta_jax, grad_thetas, dxdthetas, dx2dx1s = grad_theta(params, df, x0, us, dloss_theta)

print("Jax: ", g1)
print("Root: ", grad_theta_jax)
# print("Grad thetas: ", grad_thetas)
# print("Alternative Root Comp: ", (dxdthetas[0] * df(xs[0]) + dxdthetas[0] * dxdthetas[1] * df(xs[1]))/ 2.0)
# dtheta = dxdthetas[0] * dxdthetas[1] * df(xs[1])
# print("Chain rule computation: ", dtheta)

# num_samples=15
# us_1 = npr.rand(num_samples,2)
# us_2 = npr.rand(num_samples,2)
# x0_1 = 15.0
# x0_2 = 1.0
# xs_1, grad_theta_jax, grad_thetas, dxdthetas, dx2dx1s_1 = grad_theta(mu1, df, x0_1, us_1)
# xs_2, grad_theta_jax, grad_thetas, dxdthetas, dx2dx1s_2 = grad_theta(mu1, df, x0_2, us_2)

# xs_1 = [x0_1] + xs_1
# xs_2 = [x0_2] + xs_2
# dx2dx1s_1 = [1.0] + dx2dx1s_1
# dx2dx1s_2 = [1.0] + dx2dx1s_2
# plt.ion()
# plt.figure()
# plt.subplot(211)
# plt.plot([0, num_samples], [mu1, mu1],'k--',linewidth=0.5)
# plt.fill_between([0, num_samples], [mu1,mu1]-2*np.sqrt(sig2), [mu1,mu1]+2*np.sqrt(sig2), color='k', alpha=0.25)
# plt.plot(np.arange(0,num_samples+1), xs_1,label="run 1")
# plt.plot(np.arange(0,num_samples+1), xs_2,label="run 2")
# plt.ylabel("x")
# # plt.subplot(312)
# # plt.plot(dx2dx1s_1)
# # plt.plot(dx2dx1s_2)
# plt.subplot(212)
# plt.plot([0, num_samples], [mu1, mu1],'k--',linewidth=0.5)
# prod_dx2dx1s_1 = [np.prod(dx2dx1s_1[:i+1]) for i in range(num_samples+1)]
# prod_dx2dx1s_2 = [np.prod(dx2dx1s_2[:i+1]) for i in range(num_samples+1)]
# plt.plot(np.arange(0,num_samples+1), prod_dx2dx1s_1)
# plt.plot(np.arange(0,num_samples+1), prod_dx2dx1s_2)
# plt.ylabel("d xt / d x0")
# plt.xlabel("sample")

def loss_reparam(params, num_samples=1):
    x_new = params[0] + npr.randn(num_samples)*np.sqrt(params[1])
    return np.mean((x_new - xstar)**2) - np.log(np.sqrt(2.0 * np.pi * np.exp(1) * params[1]))
grad_reparam = grad(loss_reparam)

## Uncomment if you want to run experiment
theta_jax = np.array([-0.5,1.0])
theta_root = np.array([-0.5,1.0])
thetas_jax = []
thetas_jax.append(theta_jax)
thetas_root = []
thetas_root.append(theta_root)
theta_reparam = np.array([-0.5,1.0])
thetas_reparam = [theta_reparam]
x = [0.05]
alpha = 0.01 # learning rate
losses_jax = []
losses_root = []
num_iters=1000
num_samples=5
a0 = 0.2
gam = 0.1
# run many times
for i in range(num_iters):
    print(i)
    # draw samples
    us = npr.rand(num_samples,2)

    # learning rate
    alpha_t = a0 / (1 + gam * (i+1))

    # jax
    dfdtheta_jax = g_jax(theta_jax, x[-1], us)
    theta_jax = theta_jax - dfdtheta_jax * alpha_t    
    # print(theta_jax)
    thetas_jax.append(theta_jax)

    # root
    x, dfdtheta_root, _, _, _ = grad_theta(theta_root, df, x[-1], us, dloss_theta)
    theta_root = theta_root - dfdtheta_root * alpha_t
    thetas_root.append(theta_root)
    losses_root.append(np.mean(loss_x(np.array(x))) + loss_theta(theta_root))

    # theta reparam
    dfdtheta_reparam = grad_reparam(theta_reparam, num_samples)
    theta_reparam = theta_reparam - dfdtheta_reparam * alpha_t
    # print(theta_reparam)
    thetas_reparam.append(theta_reparam)

thetas_jax = np.array(thetas_jax)
thetas_root = np.array(thetas_root)
thetas_reparam = np.array(thetas_reparam)
plt.ion()
plt.figure()
plt.subplot(211)
# plt.title("min E[(x - 5)^2], x ~ Laplace($\mu$, 1)")
plt.plot(thetas_jax[:,0],'r',label="jax")
plt.plot(thetas_root[:,0],'b',label="root")
plt.plot(thetas_reparam[:,0],'g',label="reparam")
plt.plot(thetas_jax[:,1],'r--',label="")
plt.plot(thetas_root[:,1],'b--',label="")
plt.plot(thetas_reparam[:,1],'g--',label="")
plt.ylabel("$\mu$")
plt.plot([0,num_iters],[xstar, xstar],'k--',linewidth=0.5)
plt.subplot(212)
plt.plot(losses_root,'b', label="Slice")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.tight_layout()