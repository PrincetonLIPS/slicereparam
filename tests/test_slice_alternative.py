from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np 
from jax import jit, grad, vmap
from jax import random
from jax import lax
from jax.ops import index, index_update
from jax.flatten_util import ravel_pytree
from functools import partial

from scipy.optimize import root_scalar, brentq
import matplotlib.pyplot as plt
from tqdm.auto import trange
from jax.scipy.special import expit as sigmoid
from jax.scipy.special import logsumexp

import time

# @jit
# def f_alpha(alpha, x, d, theta, u1):
#     return log_pdf(x + alpha * d, theta) - log_pdf(x, theta) - np.log(u1)

# @jit
# def fa(x, alpha, d, theta, u1):
#     return log_pdf(x + alpha * d, theta) - log_pdf(x, theta) - np.log(u1)

@jit
def f_alpha(x, alpha, d, theta, u1):
    return log_pdf(x + alpha * d, theta) - log_pdf(x, theta) - np.log(u1)

batch_fa = jit(vmap(f_alpha, (None,0,None,None,None)))

@jit
def dual_bisect_method(
    x, d, theta, u1,
    aL=-1e5, bL=-1e-5, aR=1e-5, bR=1e5,
    tol=1e-6, maxiter=100):

    i = maxiter-1.0
    bracket_vals = [aL, bL, aR, bR]
    sign_aL = np.sign(f_alpha(x, aL, d, theta, u1))
    sign_bL = np.sign(f_alpha(x, bL, d, theta, u1))
    sign_aR = np.sign(f_alpha(x, aR, d, theta, u1))
    sign_bR = np.sign(f_alpha(x, bR, d, theta, u1))

    bracket_sign_fvals = [sign_aL, sign_bL, sign_aR, sign_bR]
    init_val = [bracket_vals, bracket_sign_fvals, i]

    def cond_fun(val):
        bracket_vals, bracket_sign_fvals, i = val
        aL, bL, aR, bR = bracket_vals 
        return np.sum(np.abs(bL-aL)) + np.sum(np.abs(bR-aR)) + 100 * np.minimum(i, 0.0) > tol

    def body_fun(val):

        # unpack val
        bracket_vals, bracket_sign_fvals, i = val
        aL, bL, aR, bR = bracket_vals 
        sign_aL, sign_bL, sign_aR, sign_bR = bracket_sign_fvals

        # new center points
        cL = (aL+bL)/2.0
        cR = (aR+bR)/2.0

        # L
        sign_cL = np.sign(f_alpha(x, cL, d, theta, u1))
        aL = lax.cond(sign_cL * sign_aL > 0, (), lambda _ : cL, (), lambda _ : aL)
        bL = lax.cond(sign_cL * sign_bL > 0, (), lambda _ : cL, (), lambda _ : bL)
        sign_aL = lax.cond(sign_cL * sign_aL > 0, (), lambda _ : sign_cL, (), lambda _ : sign_aL)
        sign_bL = lax.cond(sign_cL * sign_bL > 0, (), lambda _ : sign_cL, (), lambda _ : sign_bL)

        # R
        sign_cR = np.sign(f_alpha(x, cR, d, theta, u1))
        aR = lax.cond(sign_cR * sign_aR > 0, (), lambda _ : cR, (), lambda _ : aR)
        bR = lax.cond(sign_cR * sign_bR > 0, (), lambda _ : cR, (), lambda _ : bR)
        sign_aR = lax.cond(sign_cR * sign_aR > 0, (), lambda _ : sign_cR, (), lambda _ : sign_aR)
        sign_bR = lax.cond(sign_cR * sign_bR > 0, (), lambda _ : sign_cR, (), lambda _ : sign_bR)

        i = i - 1
        bracket_vals = [aL, bL, aR, bR]
        bracket_sign_fvals = [sign_aL, sign_bL, sign_aR, sign_bR]
        val = [bracket_vals, bracket_sign_fvals, i]

        return val

    val = lax.while_loop(cond_fun, body_fun, init_val)

    # unpack val
    bracket_vals, bracket_sign_fvals, i = val
    aL, bL, aR, bR = bracket_vals 

    # new center points
    cL = (aL+bL)/2.0
    cR = (aR+bR)/2.0

    return [cL, cR]

@jit
def dual_ridder_method(
    x, d, theta, u1,
    x1L=-1e5, x2L=-1e-5, x1R=1e-5, x2R=1e5,
    tol=1e-6, maxiter=100):

    i = maxiter-1.0
    fx1L = f_alpha(x, x1L, d, theta, u1)
    fx2L = f_alpha(x, x2L, d, theta, u1)
    fx1R = f_alpha(x, x1R, d, theta, u1)
    fx2R = f_alpha(x, x2R, d, theta, u1)
    zriddrL = 1e10
    zriddrR = -1e10
    x4L = 1e6
    x4R = -1e6
    init_val = [x1L, x2L, x1R, x2R, fx1L, fx2L, fx1R, fx2R, x4L, x4R, zriddrL, zriddrR, i]

    def cond_fun(val):
        x1L, x2L, x1R, x2R, fx1L, fx2L, fx1R, fx2R, x4L, x4R, zriddrL, zriddrR, i = val
        return np.sum(np.abs(x4L-zriddrL)) + np.sum(np.abs(x4R-zriddrR)) + 100 * np.minimum(i, 0.0) > tol

    def body_fun(val):

        x1L, x2L, x1R, x2R, fx1L, fx2L, fx1R, fx2R, x4L, x4R, zriddrL, zriddrR, i = val
        x3L = (x1L+x2L)/2.0
        x3R = (x1R+x2R)/2.0
        fx3L = f_alpha(x, x3L, d, theta, u1)
        fx3R = f_alpha(x, x3R, d, theta, u1)
        zriddrL = x4L + 0.0
        zriddrR = x4R + 0.0

        # L
        x4L = x3L + (x3L - x1L) * np.sign(fx1L-fx2L) * fx3L / np.sqrt(fx3L**2 - fx1L*fx2L)
        fx4L = f_alpha(x, x4L, d, theta, u1)

        # is cL a bracketing point?
        cond_x3L = np.maximum(-1 * np.sign(fx3L) * np.sign(fx4L), 0) # is 1 if choosing x3L, 0 if not
        cond_x1L = np.maximum(-1 * np.sign(fx1L) * np.sign(fx4L), 0) * (1 - cond_x3L) 
        cond_x2L = np.maximum(-1 * np.sign(fx2L) * np.sign(fx4L), 0) * (1 - cond_x3L) * (1 - cond_x1L)

        # set aL = cL if so, otherwise choose aL / bL that has function value opposite to fxL
        x1L = np.sum(cond_x3L * x3L + cond_x1L * x1L + cond_x2L * x4L)
        x2L = np.sum(cond_x3L * x4L + cond_x1L * x4L + cond_x2L * x2L)

        fx1L = f_alpha(x, x1L, d, theta, u1)
        fx2L = f_alpha(x, x2L, d, theta, u1)

        # R
        x4R = x3R + (x3R - x1R) * np.sign(fx1R-fx2R) * fx3R / np.sqrt(fx3R**2 - fx1R*fx2R)
        fx4R = f_alpha(x, x4R, d, theta, u1)

        # is cR a bracketing point?
        cond_x3R = np.maximum(-1 * np.sign(fx3R) * np.sign(fx4R), 0) # is 1 if choosing x3R, 0 if not
        cond_x1R = np.maximum(-1 * np.sign(fx1R) * np.sign(fx4R), 0) * (1 - cond_x3R) 
        cond_x2R = np.maximum(-1 * np.sign(fx2R) * np.sign(fx4R), 0) * (1 - cond_x3R) * (1 - cond_x1R)

        # set aR = cR if so, otherwise choose aR / bR that has function vaRue opposite to fxR
        x1R = np.sum(cond_x3R * x3R + cond_x1R * x1R + cond_x2R * x4R)
        x2R = np.sum(cond_x3R * x4R + cond_x1R * x4R + cond_x2R * x2R)

        fx1R = f_alpha(x, x1R, d, theta, u1)
        fx2R = f_alpha(x, x2R, d, theta, u1)

        i = i - 1
        val = [x1L, x2L, x1R, x2R, fx1L, fx2L, fx1R, fx2R, x4L, x4R, zriddrL, zriddrR, i]

        return val

    val = lax.while_loop(cond_fun, body_fun, init_val)
    x1L, x2L, x1R, x2R, fx1L, fx2L, fx1R, fx2R, x4L, x4R, zriddrL, zriddrR, i = val 
    return [zriddrL, zriddrR]

@jit
def brent_method(
    x, d, theta, u1, a, b, tol=1e-6, maxiter=100):
    # f(a) < f(b) is important 

    i = maxiter-1.0
    c = a + 0.0
    s = 0.0 # initialize
    dd = 0.0
    mflag = 1.0
    fa, fb, fc = batch_fa(x, np.array([a,b,c]), d, theta, u1)
    init_val = [a, b, c, fa, fb, fc, dd, mflag, s, i]

    def iqi(a, b, c, fa, fb, fc):
        s =     a * fb * fc / (fa - fb) / (fa - fc)
        s = s + b * fa * fc / (fb - fa) / (fb - fc)
        s = s + c * fa * fb / (fc - fa) / (fc - fb)
        return s

    def secant(a, b, fa, fb):
        return b - fb * (b - a) / (fb - fa)

    # def bisect_pred(s, a, b, c, fa, fb, fc, mflag):
    #     cond1 = s < (3*a+b)/4.0 and s > b 
    #     cond2 = mflag and np.abs(s-b) >= np.abs(b-c)/2.0
    #     cond3 = not mflag and np.abs(s-b) >= np.abs(c-d)/2.0
    #     cond4 = mflag and np.abs(b-c) < tol 
    #     cond5 = not mflag and np.abs(c-d) < tol
    #     return cond1 or cond2 or cond3 or cond4 or cond5

    def bisect_pred(s, a, b, c, fa, fb, fc, dd, mflag):
        # want things to be positive numbers if true
        cond1 = np.sign((3*a+b)/4.0 - s) + np.sign(s - b) - 0.1 # >0 only if both signs are +1
        cond2 = mflag * np.sign( np.abs(s-b) - np.abs(b-c)/2.0 + 1e-15)
        cond3 = (1.0 - mflag) * np.sign( np.abs(s-b) - np.abs(c-dd)/2.0 + 1e-15)
        cond4 = mflag * np.sign(tol - np.abs(b-c))
        cond5 = (1.0 - mflag) * np.sign(tol - np.abs(c-dd))
        out = np.maximum(cond1, 0.0) + np.maximum(cond2, 0.0) \
            + np.maximum(cond3, 0.0) + np.maximum(cond4, 0.0) + np.maximum(cond5, 0.0)
        return np.sum(out)

    # mflag = 1.0

    # def iqi_pred(fa, fb, fc):
    #     cond1 = np.abs(fa - fc)
    #     cond1 = not fa == fc
    #     cond2 = not fb == fc
    #     return cond1 and cond2

    def cond_fun(val):
        a, b, c, fa, fb, fc, dd, mflag, s, i = val
        return np.sum(np.abs(b-a)) + 100 * np.minimum(i, 0.0) > tol

    def body_fun(val):
        a, b, c, fa, fb, fc, dd, mflag, s, i = val

        # step 1: iqi or secant
        # s = lax.cond(np.abs(fa - fc) > 1e-6, (), lambda _ : iqi(a, b, c, fa, fb, fc), (), lambda _ : secant(a, b, fa, fb))
        # s = lax.cond(np.abs(fb - fc) > 1e-6, (), lambda _ : iqi(a, b, c, fa, fb, fc), (), lambda _ : secant(a, b, fa, fb))
        iqi_flag = 1.0 / np.abs(fa - fc) + 1.0 / np.abs(fb - fc)
        s = lax.cond(iqi_flag < 1e8, (), lambda _ : iqi(a, b, c, fa, fb, fc), (), lambda _ : secant(a, b, fa, fb))

        # step 2 : bisect?
        out = bisect_pred(s, a, b, c, fa, fb, fc, dd, mflag)
        s = lax.cond(out > tol, (), lambda _ : (a+b)/2.0, (), lambda _ : s)
        mflag = lax.cond(out > tol, (), lambda _ : 1.0, (), lambda _ : 0.0)

        fs = f_alpha(x, s, d, theta, u1)
        dd = c + 0.0
        c = b + 0.0
        fc = f_alpha(x, c, d, theta, u1)
        
        b = lax.cond(fa * fs < 0, (), lambda _ : s, (), lambda _ : b) 
        a = lax.cond(fa * fs < 0, (), lambda _ : a, (), lambda _ : s) 

        fa, fb = batch_fa(x, np.array([a, b]), d, theta, u1)
        a_new = lax.cond(np.abs(fa) < np.abs(fb), (), lambda _ : b, (), lambda _ : a)
        b_new = lax.cond(np.abs(fa) < np.abs(fb), (), lambda _ : a, (), lambda _ : b)
        a = a_new + 0.0 
        b = b_new + 0.0
        fa, fb = batch_fa(x, np.array([a, b]), d, theta, u1)

        i = i - 1.0

        val = [a, b, c, fa, fb, fc, dd, mflag, s, i]

        return val

    val = lax.while_loop(cond_fun, body_fun, init_val)
    a, b, c, fa, fb, fc, dd, mflag, s, i = val
    return (b+a)/2.0

dual_brent_method = jit(vmap(brent_method, (None, None, None, None, 0, 0)))
# a_grid = np.concatenate((np.logspace(-3,1, 25), np.array([25.0])))
@jit
def choose_start(
    x, d, theta, u1,
    log_start = -3.0, log_space = 1.0 / 6.0):

    i = 0
    aL = -1.0 * np.power(10.0, log_start + i * log_space)
    bR = np.power(10.0, log_start + i * log_space)
    aL_val, bR_val = batch_fa(x, np.array([aL, bR]), d, theta, u1)
    init_val = [aL, bR, aL_val, bR_val, i]

    def cond_fun(val):
        aL, bR, aL_val, bR_val, i = val
        return np.maximum(aL_val, 0.0) + np.maximum(bR_val, 0.0) > 0.0

    def body_fun(val):

        aL, bR, aL_val, bR_val, i = val
        i = i+1
        sign_aL = np.sign(aL_val)
        sign_bR = np.sign(bR_val)
        aL = np.sum(-1.0 * np.power(10.0, log_start + i * log_space) * np.maximum(sign_aL, 0.0) \
                + aL * np.maximum(-1.0 * sign_aL, 0.0))
        bR = np.sum(np.power(10.0, log_start + i * log_space) * np.maximum(sign_bR, 0.0) \
                + bR * np.maximum(-1.0 * sign_bR, 0.0))
        aL_val, bR_val = batch_fa(x, np.array([aL, bR]), d, theta, u1)
        val = [aL, bR, aL_val, bR_val, i]

        return val

    val = lax.while_loop(cond_fun, body_fun, init_val)
    aL, bR, aL_val, bR_val, i = val
    return [aL, bR]

@jit
def forwards_step2(x, theta, u1, u2, d):#, aL, bR):
    aL, bR = choose_start(x, d, theta, u1)
    # z_L, z_R = dual_bisect_method(x, d, theta, u1, aL=aL, bL=-1e-10, aR=1e-10, bR=bR)
    # z_L, z_R = dual_ridder_method(x, d, theta, u1, x1L=aL, x2L=-1e-10, x1R=1e-10, x2R=bR)
    # z_L = brent_method(x, d, theta, u1, aL, -1e-10)
    # z_R = brent_method(x, d, theta, u1, 1e-10, bR)
    a = np.array([aL, 1e-10])
    b = np.array([-1e-10, bR])
    z_L, z_R = dual_brent_method(x, d, theta, u1, a, b)
    x_L = x + d*z_L
    x_R = x + d*z_R
    x = (1 - u2) * x_L + u2 * x_R
    alphas = np.array([z_L, z_R])
    return x, x_L, x_R, alphas

@jit
def forwards_step(x, theta, u1, u2, d):#, aL, bR):
    aL, bR = choose_start(x, d, theta, u1)
    z_L, z_R = dual_bisect_method(x, d, theta, u1, aL=aL, bL=-1e-10, aR=1e-10, bR=bR)
    x_L = x + d*z_L
    x_R = x + d*z_R
    x = (1 - u2) * x_L + u2 * x_R
    alphas = np.array([z_L, z_R])
    return x, x_L, x_R, alphas

vmapped_forwards_step = jit(vmap(forwards_step, (0,None,0,0,0)))
vmapped_forwards_step2 = jit(vmap(forwards_step2, (0,None,0,0,0)))

# @jit

def forwards(S, theta, x, us, ds):
    xs = [x]
    xLs = []
    xRs = []
    alphas = []
    for s in range(S):
        # x, x_L, x_R, alpha = forwards_step(x, theta, us[s,:,0], us[s,:,1], ds[s])#, aL, bR)
        x, x_L, x_R, alpha = vmapped_forwards_step(x, theta, us[s,:,0], us[s,:,1], ds[s])#, aL, bR)
        xs.append(x)
        xLs.append(x_L)
        xRs.append(x_R)
        alphas.append(alpha)
    return np.array(xs), np.array(xLs), np.array(xRs), np.array(alphas)

@partial(jit, static_argnums={0})
def jit_forwards(S, theta, x, us, ds):
    xs = np.zeros((S+1, num_chains, D))
    xs = index_update(xs, index[0, :, :], x)
    xLs = np.zeros((S, num_chains, D))
    xRs = np.zeros((S, num_chains, D))
    alphas = np.zeros((S, num_chains, 2))
    init_val = [xs, xLs, xRs, alphas, x]

    def body_fun(i, val):
        xs, xLs, xRs, alphas, x = val 
        x, x_L, x_R, alpha = vmapped_forwards_step(x, theta, us[i,:,0], us[i,:,1], ds[i])
        xs = index_update(xs, index[i+1, :, :], x)
        xLs = index_update(xLs, index[i, :, :], x_L)
        xRs = index_update(xRs, index[i, :, :], x_R)
        alphas = index_update(alphas, index[i, :, :], alpha)
        val = [xs, xLs, xRs, alphas, x]
        return val

    xs, xLs, xRs, alphas, x = lax.fori_loop(0, S, body_fun, init_val)
    return xs, xLs, xRs, alphas


def forwards2(S, theta, x, us, ds):
    xs = [x]
    xLs = []
    xRs = []
    alphas = []
    for s in range(S):
        # x, x_L, x_R, alpha = forwards_step(x, theta, us[s,:,0], us[s,:,1], ds[s])#, aL, bR)
        x, x_L, x_R, alpha = vmapped_forwards_step2(x, theta, us[s,:,0], us[s,:,1], ds[s])#, aL, bR)
        xs.append(x)
        xLs.append(x_L)
        xRs.append(x_R)
        alphas.append(alpha)
    return np.array(xs), np.array(xLs), np.array(xRs), np.array(alphas)


var1 = 1.0
var2 = 2.0
def _log_pdf(x, params):

    mu1 = params[0]
    mu2 = params[1]

    log1 = -0.5 * (x - mu1)**2 / var1 - 0.5 * np.sqrt(2.0 * np.pi * var1)
    log2 = -0.5 * (x - mu2)**2 / var2 - 0.5 * np.sqrt(2.0 * np.pi * var2)
    return np.sum(logsumexp(np.array([log1,log2]),axis=0))

# compute necessary gradients
def log_pdf_theta(theta, x):    return log_pdf(x, theta)
def log_pdf_x(x, theta):        return log_pdf(x, theta)
def log_pdf_ad(x, theta, a, d): return log_pdf(x + a * d, theta)
grad_x = jit(grad(log_pdf_x))
grad_theta = jit(grad(log_pdf_theta))
grad_x_ad = jit(grad(log_pdf_ad))

_params = [-3.0, 3.0]
params, unflatten = ravel_pytree(_params)
log_pdf = jit(lambda x, params : _log_pdf(x, unflatten(params)))

D = 1
key = random.PRNGKey(4)

S = 200 # number of samples
num_chains = 100 # number of chains
key, *subkeys = random.split(key, 4)
us = random.uniform(subkeys[0], (S,num_chains,2))
ds = random.normal(subkeys[1], (S*num_chains,D))
ds_norm = ds / np.sqrt(np.sum(ds**2, axis=1))[:,None]
ds_norm = ds_norm.reshape((S, num_chains, D))
x = 1.0 * random.normal(subkeys[2], (num_chains,D)) # initial x 

# run forward pass
t1 = time.time()
xs, xLs, xRs, alphas = forwards(S, params, x, us, ds_norm)
t2 = time.time()
print(t2-t1)
t1 = time.time()
xs2, xLs2, xRs2, alphas2 = jit_forwards(S, params, x, us, ds_norm)
t2 = time.time()
print(t2-t1)
print(np.linalg.norm(xs - xs2))
print(np.linalg.norm(xLs - xLs2))
print(np.linalg.norm(xRs - xRs2))
print(np.linalg.norm(alphas - alphas2))

# t1 = time.time()
# xs2, xLs2, xRs2, alphas2 = forwards2(S, params, x, us, ds_norm)
# t2 = time.time()
# print(t2-t1)
xs_plot = np.reshape(xs[1:,:,0], S*num_chains)
xs_plot2 = np.reshape(xs2[1:,:,0], S*num_chains)
dx = 0.01
x_range = np.arange(-10,10,dx)
# g1 = 1.0 / np.sqrt(2.0 * np.pi * var1) * np.exp(-0.5 * (x_range - _params[0])**2 / var1)
# g2 = 1.0 / np.sqrt(2.0 * np.pi * var2) * np.exp(-0.5 * (x_range - _params[1])**2 / var2)
# z1 = 0.5 

pdf = np.array([np.exp(log_pdf(x, params)) for x in x_range])
normalizer = np.sum(pdf)*dx

plt.figure(figsize=[8,4])
plt.subplot(211)
plt.plot(x_range, pdf / normalizer)
plt.hist(xs_plot, 80, density=True);
plt.title("with step out bracket")

plt.subplot(212)
plt.plot(x_range, pdf / normalizer)
plt.hist(xs_plot2, 80, density=True);
plt.title("with step out bracket")

plt.figure()
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.plot(x_range, pdf / normalizer)
    plt.hist(xs2[1:,i,0], 20, density=True)
    plt.xticks([])
    plt.yticks([])

# @jit
# def forwards_step_without_choose_start(x, theta, u1, u2, d):#, aL, bR):
#     z_L, z_R = dual_bisect_method(x, d, theta, u1, bL=-1e-10, aR=1e-10)
#     x_L = x + d*z_L
#     x_R = x + d*z_R
#     x = (1 - u2) * x_L + u2 * x_R
#     alphas = np.array([z_L, z_R])
#     return x, x_L, x_R, alphas

# vmapped_forwards_step_without_choose_start = jit(vmap(forwards_step_without_choose_start, (0,None,0,0,0)))

# # @jit
# def forwards_without_choose_start(S, theta, x, us, ds):
#     xs = [x]
#     xLs = []
#     xRs = []
#     alphas = []
#     for s in range(S):
#         x, x_L, x_R, alpha = vmapped_forwards_step_without_choose_start(x, theta, us[s,:,0], us[s,:,1], ds[s])#, aL, bR)
#         xs.append(x)
#         xLs.append(x_L)
#         xRs.append(x_R)
#         alphas.append(alpha)
#     return np.array(xs), np.array(xLs), np.array(xRs), np.array(alphas)

# # run forward pass
# xs2, xLs, xRs, alphas = forwards_without_choose_start(S, params, x, us, ds_norm)
# xs_plot2 = np.reshape(xs2[1:,:,0], S*num_chains)
# # plt.figure(figsize=[8,4])
# plt.subplot(212)
# plt.plot(x_range, pdf / normalizer)
# plt.hist(xs_plot2, 80, density=True, label="w/o choose start");
# plt.title("without step out bracket")

# plt.figure()
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.plot(x_range, pdf / normalizer)
#     plt.hist(xs[1:,i,0], 20, density=True)
#     plt.xticks([])
#     plt.yticks([])


# ## backwrads sample
# @jit
# def backwards_step(theta, dL_dtheta, us, d, x, xL, xR, alphas, dL_dx, prev_dL_dx):

#     u1 = us[0]
#     u2 = us[1]
#     z_L = alphas[0]
#     z_R = alphas[1]

#     # compute loss for current sample
#     # set prev_dL_dx to zero at first
#     dL_dx_s = dL_dx + prev_dL_dx

#     # compute gradients of xL and xR wrt theta
#     L_grad_theta = -1.0 * (grad_theta(theta, xL) - grad_theta(theta, x)) / np.dot(d, grad_x_ad(x, theta, z_L, d))
#     R_grad_theta = -1.0 * (grad_theta(theta, xR) - grad_theta(theta, x)) / np.dot(d, grad_x_ad(x, theta, z_R, d))

#     # compute gradient dL / dtheta
#     dLd = np.dot(dL_dx_s, d) # dot product between loss gradient and direction - this is used multiple times 
#     dL_dtheta_s = u2 * dLd * R_grad_theta + (1-u2) * dLd * L_grad_theta
#     dL_dtheta = dL_dtheta + dL_dtheta_s

#     # propagate loss backwards : compute gradient times Jacobian of dx_s  / dx_{s-1}
#     L_grad_x = -1.0 * ( grad_x_ad(x, theta, z_L, d) - grad_x(x, theta) ) / np.dot(d, grad_x_ad(x, theta, z_L, d))
#     R_grad_x = -1.0 * ( grad_x_ad(x, theta, z_R, d) - grad_x(x, theta) ) / np.dot(d, grad_x_ad(x, theta, z_R, d))
#     prev_dL_dx = dL_dx_s + u2 * dLd * R_grad_x + (1-u2) * dLd * L_grad_x

#     J_x = np.eye(D) + u2 * np.outer(d, R_grad_x) + (1-u2) * np.outer(d, L_grad_x)

#     return dL_dtheta, prev_dL_dx, J_x

# def get_jacobians(S, theta, us, ds, xs, xLs, xRs, alphas, dL_dxs):
#     dL_dtheta = np.zeros_like(theta)
#     prev_dL_dx = np.zeros_like(xs[0])
#     J_xs = []
#     for s in range(S-1, -1, -1):
#         dL_dtheta, prev_dL_dx, J_x = backwards_step(theta, dL_dtheta, us[s,:], ds[s], xs[s], 
#                                                xLs[s], xRs[s], alphas[s], dL_dxs[s], prev_dL_dx)
#         J_xs.append(J_x)
#     return dL_dtheta, J_xs


# S = 100 # number of samples
# num_chains = 1 # number of chains
# key, *subkeys = random.split(key, 4)
# us = random.uniform(subkeys[0], (S,num_chains,2))
# ds = random.normal(subkeys[1], (S*num_chains,D))
# ds_norm = ds / np.sqrt(np.sum(ds**2, axis=1))[:,None]
# ds_norm = ds_norm.reshape((S, num_chains, D))
# x = 0.0 + 3.0 * random.normal(subkeys[2], (num_chains,D)) # initial x 
# xs, xLs, xRs, alphas = forwards(S, params, x, us, ds_norm)

# chain_number = 0
# dL_dtheta, J_xs = get_jacobians(S, params, us[:,chain_number,:], ds_norm[:,chain_number,:],
#                                 xs[:,chain_number,:], xLs[:,chain_number,:], xRs[:,chain_number,:],
#                                 alphas[:,chain_number,:], np.zeros((S,D)))
# J_xs.reverse()
# J_xs = np.array(J_xs)
# S_plot = 100
# plt.figure()
# plt.subplot(211)
# plt.plot(np.arange(S_plot+1), xs[:S_plot+1,chain_number,0])
# plt.ylabel("$x_n$")
# # plt.subplot(312)
# # plt.plot(np.arange(S_plot)+1, np.abs((J_xs[:S_plot,0])))
# # plt.xlabel("iteration")
# plt.subplot(212)
# plt.plot(np.arange(S_plot)+1, np.abs(np.cumprod(J_xs[:S_plot,0])))
# plt.xlabel("iteration n")
# plt.ylabel("$| dx_n / dx_0 |$")