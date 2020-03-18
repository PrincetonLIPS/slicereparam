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


mu = np.zeros(2)
Sigma = 1*np.eye(2)
# mu = np.array([-0.5,0.5])
# Sigma = np.array([[1.0, 0.9], [0.9,1.0]])
# Sigma = np.eye()
# def pdf(x):
#     d = x.shape[0]
#     normalizer =  np.power(2*np.pi, d/2) * np.sqrt(np.linalg.det(Sigma))
#     return np.exp(-0.5 * (x - mu).T @ np.linalg.inv(Sigma) @ (x - mu)) / normalizer
# def pdf(x):
    # return -x[0]**2 - 3.0*np.abs(x[1])
def pdf(x):
    return np.exp(-np.abs(x[0]) / 1.0 - np.abs(x[1])/ 1.0 -x[0]**2 / 1.0 -x[1]**2/1.0 )

w1 = 0.5
w2 = 0.5
mu1 = np.zeros(2)
Sigma1 = 1*np.eye(2)
mu2 = np.array([-0.5,0.5])
Sigma2 = np.array([[1.0, 0.7], [0.7,1.0]])
def pdf(x):
    d = x.shape[0]
    normalizer1 =  np.power(2*np.pi, d/2) * np.sqrt(np.linalg.det(Sigma1))
    p1 = np.exp(-0.5 * (x - mu1).T @ np.linalg.inv(Sigma1) @ (x - mu1)) / normalizer1
    normalizer2 =  np.power(2*np.pi, d/2) * np.sqrt(np.linalg.det(Sigma2))
    p2 = np.exp(-0.5 * (x - mu2).T @ np.linalg.inv(Sigma2) @ (x - mu2)) / normalizer2
    return w1*p1 + w2*p2

w1 = 0.5
w2 = 0.5
w3 = 0.5
w4 = 0.5
mu1 = np.array([-1,-1])
mu2 = np.array([1,-1])
mu3 = np.array([-1,1])
mu4 = np.array([1,1])
Sigma = 1*np.eye(2)
def pdf(x):
    d = x.shape[0]
    p1 = np.exp(-0.5 * np.dot(x - mu1, x - mu1) / 0.5)
    p2 = np.exp(-0.5 * np.dot(x - mu2, x - mu2) / 0.5)
    p3 = np.exp(-0.5 * np.dot(x - mu3, x - mu3) / 0.5)
    p4 = np.exp(-0.5 * np.dot(x - mu4, x - mu4) / 0.5)
    return w1*p1 + w2*p2 + w3*p3 + w4*p4

# H = 25
# scale = 0.1
# W1 = scale*npr.randn(H,2)
# b1 = scale/2*npr.randn(H)
# W2 = scale*npr.randn(H,H)
# b2 = scale/2*npr.randn(H)
# W3 = scale*npr.randn(1,H)
# def relu(x):    return np.maximum(0, x)
# def pdf(x):
#     h1 = relu(W1@x + b1)
#     h2 = relu(W2@h1 + b2)
#     out = W3@h2
#     d = x.shape[0]
#     normalizer =  np.power(2*np.pi, d/2) * np.sqrt(np.linalg.det(Sigma))
#     return  np.exp(out) * np.exp(-0.5 * (x - mu).T @ np.linalg.inv(Sigma) @ (x - mu)) / normalizer

# run slice sampler
S = 10000
x = npr.randn(2)
xs = [x]

for s in range(S):

    # sample x1
    if np.mod(s, 50) == 0:
        print(s)

    u1 = npr.rand()
    u2 = npr.rand()
    y = pdf(x) * u1
    fx = lambda x1 : pdf(np.array([x1, x[1]])) - y
    x0 = np.copy(x[0])
    res_L = root_scalar(fx, x0=x0-1e-3, method="brentq", bracket=[-1e8,x0-1e-5])
    res_R = root_scalar(fx, x0=x0+1e-3, method="brentq", bracket=[x0+1e-5,1e8])
    x_L = res_L.root
    x_R = res_R.root
    x1 = (1 - u2) * x_L + u2 * x_R
    x = np.array([x1, x[1]])

    # sample x2
    u1 = npr.rand()
    u2 = npr.rand()
    y = pdf(x) * u1
    fx = lambda x2 : pdf(np.array([x[0], x2])) - y
    x0 = np.copy(x[1])
    res_L = root_scalar(fx, x0=x0-1e-3, method="brentq", bracket=[-1e8,x0-1e-5])
    res_R = root_scalar(fx, x0=x0+1e-3, method="brentq", bracket=[x0+1e-5,1e8])
    x_L = res_L.root
    x_R = res_R.root
    x2 = (1 - u2) * x_L + u2 * x_R
    x = np.array([x[0], x2])
    
    xs.append(x)

xmin = -3
xmax = 3
dx=0.1

xs_plot = np.array(xs)
plt.ion()
plt.figure(figsize=[8,4])
plt.subplot(121)
plt.hist2d(xs_plot[:,0], xs_plot[:,1], range=[[xmin, xmax],[xmin, xmax]], bins=25, density=True, vmin=0.0, vmax=0.1)
plt.xlim([xmin, xmax])
plt.ylim([xmin, xmax])
plt.colorbar()

# xs2 = npr.multivariate_normal(mu, Sigma, S)
# plt.figure()
# plt.hist2d(xs2[:,0], xs2[:,1], bins=25)
plt.subplot(122)
x1 = np.arange(xmin,xmax+dx,dx)
x2 = np.arange(xmin,xmax+dx,dx)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros(X1.shape)
for i in range(x1.shape[0]):
    for j in range(x2.shape[0]):
        Z[j,i] = pdf(np.array([x1[i], x2[j]]))
plt.imshow(Z / (np.sum(Z) * dx**2), extent=[xmin,xmax,xmin,xmax], origin="lower", vmin=0.0, vmax=0.1)
# plt.plot(mu[0], mu[1], 'k*')
plt.xlim([xmin, xmax])
plt.ylim([xmin, xmax])
plt.colorbar()
plt.tight_layout()