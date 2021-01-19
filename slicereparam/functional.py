from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, grad, vmap
from jax import random
from jax import lax
from jax import custom_vjp
from jax.ops import index, index_update
from jax.flatten_util import ravel_pytree
from functools import partial

from slicereparam.rootfinder import dual_bisect_method, choose_start

from inspect import signature
import warnings 


def setup_slice_sampler(log_pdf, D, S, num_chains=1):
    """This function takes as input the log pdf, parameters.
        It returns a differentiable slice sampling function (using custom vjp).
        The function generates (S) samples from (num_chains) number of chains."""
    
    # set up for backwards pass
    # compute necessary gradients
    # TODO - modify code so log_pdf is always called in same order (fix the theta switch, just take grad differently).
    def log_pdf_theta(theta, x):    return log_pdf(x, theta)
    def log_pdf_x(x, theta):        return log_pdf(x, theta)
    def log_pdf_ad(x, theta, a, d): return log_pdf(x + a * d, theta)
    grad_x = jit(grad(log_pdf_x))
    grad_theta = jit(grad(log_pdf_theta))
    grad_x_ad = jit(grad(log_pdf_ad))

    def forwards_step(x, theta, u1, u2, d):#, aL, bR):
        func = lambda alpha : log_pdf(x + alpha * d, theta) - log_pdf(x, theta) - jnp.log(u1) # root
        aL, bR = choose_start(func)
        z_L, z_R = dual_bisect_method(func, aL=aL, bL=-1e-10, aR=1e-10, bR=bR)
        x_L = x + d*z_L
        x_R = x + d*z_R
        x = (1 - u2) * x_L + u2 * x_R
        alphas = jnp.array([z_L, z_R])
        return x, x_L, x_R, alphas

    def forwards_sample(theta, x0, key):
        # generate randomness 
        key, *subkeys = random.split(key, 3)
        us = random.uniform(subkeys[0], (num_chains, S, 2))
        ds_unnorm = random.normal(subkeys[1], (S * num_chains, D))
        ds = ds_unnorm / jnp.sqrt(jnp.sum(ds_unnorm**2, axis=1))[:,None]
        ds = ds.reshape((num_chains, S, D))

        xs = jnp.zeros((num_chains, S+1, D))
        xs = index_update(xs, index[:, 0, :], x0)
        xLs = jnp.zeros((num_chains, S, D))
        xRs = jnp.zeros((num_chains, S, D))
        alphas = jnp.zeros((num_chains, S, 2))
        init_val = [xs, xLs, xRs, alphas, x0]

        def body_fun(i, val):
            xs, xLs, xRs, alphas, x = val 
            x, x_L, x_R, alpha = vmap(forwards_step, (0,None,0,0,0))(x, theta, us[:,i,0], us[:,i,1], ds[:,i,:])
            xs = index_update(xs, index[:, i+1, :], x)
            xLs = index_update(xLs, index[:, i, :], x_L)
            xRs = index_update(xRs, index[:, i, :], x_R)
            alphas = index_update(alphas, index[:, i, :], alpha)
            val = [xs, xLs, xRs, alphas, x]
            return val

        xs, xLs, xRs, alphas, x = lax.fori_loop(0, S, body_fun, init_val)
        return xs, us, ds, xLs, xRs, alphas

    def backwards_step(theta, dL_dtheta, us, d, x, xL, xR, alphas, dL_dx, prev_dL_dx):

        u1 = us[0]
        u2 = us[1]
        z_L = alphas[0]
        z_R = alphas[1]

        # compute loss for current sample
        # set prev_dL_dx to zero at first
        dL_dx_s = dL_dx + prev_dL_dx

        # compute gradients of xL and xR wrt theta
        L_grad_theta = -1.0 * (grad_theta(theta, xL) - grad_theta(theta, x)) / jnp.dot(d, grad_x_ad(x, theta, z_L, d))
        R_grad_theta = -1.0 * (grad_theta(theta, xR) - grad_theta(theta, x)) / jnp.dot(d, grad_x_ad(x, theta, z_R, d))

        # compute gradient dL / dtheta
        dLd = jnp.dot(dL_dx_s, d) # dot product between loss gradient and direction - this is used multiple times 
        dL_dtheta_s = u2 * dLd * R_grad_theta + (1-u2) * dLd * L_grad_theta
        dL_dtheta = dL_dtheta + dL_dtheta_s

        # propagate loss backwards : compute gradient times Jacobian of dx_s  / dx_{s-1}
        L_grad_x = -1.0 * ( grad_x_ad(x, theta, z_L, d) - grad_x(x, theta) ) / jnp.dot(d, grad_x_ad(x, theta, z_L, d))
        R_grad_x = -1.0 * ( grad_x_ad(x, theta, z_R, d) - grad_x(x, theta) ) / jnp.dot(d, grad_x_ad(x, theta, z_R, d))
        prev_dL_dx = dL_dx_s + u2 * dLd * R_grad_x + (1-u2) * dLd * L_grad_x

        return dL_dtheta, prev_dL_dx

    def backwards(S, theta, us, ds, xs, xLs, xRs, alphas, dL_dxs):

        dL_dtheta = jnp.zeros_like(theta)
        prev_dL_dx = jnp.zeros_like(xs[0])
        init_val = [S-1, dL_dtheta, prev_dL_dx]

        def cond_fun(val):
            return val[0] > -1

        def body_fun(val):
            s = val[0]
            dL_dtheta, prev_dL_dx = val[1:] 
            dL_dtheta, prev_dL_dx = backwards_step(theta, dL_dtheta, us[s,:], ds[s], xs[s], 
                                                xLs[s], xRs[s], alphas[s], dL_dxs[s], prev_dL_dx)
            val[0] -= 1
            return [val[0], dL_dtheta, prev_dL_dx]

        val = lax.while_loop(cond_fun, body_fun, init_val)
        dL_dtheta, prev_dL_dx = val[1:]
        return dL_dtheta, prev_dL_dx

    vmapped_backwards = vmap(backwards, (None, None, 0, 0, 0, 0, 0, 0, 0))

    @custom_vjp
    def slice_sample(theta, x0, key):
        forwards_out = forwards_sample(theta, x0, key)
        xs = forwards_out[0][:, 1:, :] # return all samples except initial condition
        return xs 

    def slice_sample_fwd(theta, x0, key):
        forwards_out = forwards_sample(theta, x0, key)
        xs = forwards_out[0][:, 1:, :] # return all samples except initial condition
        return xs, (forwards_out, theta)

    def slice_sample_bwd(res, g):
        # g has size of xs in slice sample
        # grad theta, needs to be size of theta
        # grad_x0 , needs to be size of x0
        forwards_out, theta = res
        xs0, us, ds, xLs, xRs, alphas = forwards_out
        grad_thetas, grad_x0 = vmapped_backwards(
            S, theta, us, ds, xs0, xLs, xRs, alphas, g)
        grad_theta = jnp.sum(grad_thetas, axis=0)
        return (grad_theta, grad_x0, None)

    slice_sample.defvjp(slice_sample_fwd, slice_sample_bwd)
    slice_sample = jit(slice_sample)

    return slice_sample


def setup_slice_sampler_with_args(log_pdf, D, S, num_chains=1):
    """This function takes as input the log pdf, parameters.
        It returns a differentiable slice sampling function (using custom vjp).
        The function generates (S) samples from (num_chains) number of chains.
        In this case, the log pdf takes a third argument.
        
        log_pdf(x, theta, y)
        """
    
    # set up for backwards pass
    # compute necessary gradients
    grad_x = jit(grad(log_pdf, argnums=0))
    grad_theta = jit(grad(log_pdf, argnums=1))
    def log_pdf_ad(x, theta, a, d, y): return log_pdf(x + a * d, theta, y)
    grad_x_ad = jit(grad(log_pdf_ad))

    def forwards_step(x, theta, u1, u2, d, y):
        func = lambda alpha : log_pdf(x + alpha * d, theta, y) - log_pdf(x, theta, y) - jnp.log(u1) # root
        aL, bR = choose_start(func)
        z_L, z_R = dual_bisect_method(func, aL=aL, bL=-1e-10, aR=1e-10, bR=bR)
        x_L = x + d*z_L
        x_R = x + d*z_R
        x = (1 - u2) * x_L + u2 * x_R
        alphas = jnp.array([z_L, z_R])
        return x, x_L, x_R, alphas

    def forwards_sample(theta, x0, ys, key):
        # generate randomness 
        key, *subkeys = random.split(key, 3)
        us = random.uniform(subkeys[0], (num_chains, S, 2))
        ds_unnorm = random.normal(subkeys[1], (S * num_chains, D))
        ds = ds_unnorm / jnp.sqrt(jnp.sum(ds_unnorm**2, axis=1))[:,None]
        ds = ds.reshape((num_chains, S, D))

        xs = jnp.zeros((num_chains, S+1, D))
        xs = index_update(xs, index[:, 0, :], x0)
        xLs = jnp.zeros((num_chains, S, D))
        xRs = jnp.zeros((num_chains, S, D))
        alphas = jnp.zeros((num_chains, S, 2))
        init_val = [xs, xLs, xRs, alphas, x0]

        def body_fun(i, val):
            xs, xLs, xRs, alphas, x = val 
            x, x_L, x_R, alpha = vmap(forwards_step, (0,None,0,0,0,0))(x, theta, us[:,i,0], us[:,i,1], ds[:,i,:], ys)
            xs = index_update(xs, index[:, i+1, :], x)
            xLs = index_update(xLs, index[:, i, :], x_L)
            xRs = index_update(xRs, index[:, i, :], x_R)
            alphas = index_update(alphas, index[:, i, :], alpha)
            val = [xs, xLs, xRs, alphas, x]
            return val

        xs, xLs, xRs, alphas, x = lax.fori_loop(0, S, body_fun, init_val)
        return xs, us, ds, xLs, xRs, alphas

    def backwards_step(theta, dL_dtheta, us, d, x, xL, xR, alphas, dL_dx, prev_dL_dx, y):

        u1 = us[0]
        u2 = us[1]
        z_L = alphas[0]
        z_R = alphas[1]

        # compute loss for current sample
        # set prev_dL_dx to zero at first
        dL_dx_s = dL_dx + prev_dL_dx

        # compute gradients of xL and xR wrt theta
        L_grad_theta = -1.0 * (grad_theta(xL, theta, y) - grad_theta(x, theta, y)) / jnp.dot(d, grad_x_ad(x, theta, z_L, d, y))
        R_grad_theta = -1.0 * (grad_theta(xR, theta, y) - grad_theta(x, theta, y)) / jnp.dot(d, grad_x_ad(x, theta, z_R, d, y))

        # compute gradient dL / dtheta
        dLd = jnp.dot(dL_dx_s, d) # dot product between loss gradient and direction - this is used multiple times 
        dL_dtheta_s = u2 * dLd * R_grad_theta + (1-u2) * dLd * L_grad_theta
        dL_dtheta = dL_dtheta + dL_dtheta_s

        # propagate loss backwards : compute gradient times Jacobian of dx_s  / dx_{s-1}
        L_grad_x = -1.0 * ( grad_x_ad(x, theta, z_L, d, y) - grad_x(x, theta, y) ) / jnp.dot(d, grad_x_ad(x, theta, z_L, d, y))
        R_grad_x = -1.0 * ( grad_x_ad(x, theta, z_R, d, y) - grad_x(x, theta, y) ) / jnp.dot(d, grad_x_ad(x, theta, z_R, d, y))
        prev_dL_dx = dL_dx_s + u2 * dLd * R_grad_x + (1-u2) * dLd * L_grad_x

        return dL_dtheta, prev_dL_dx

    def backwards(S, theta, us, ds, xs, xLs, xRs, alphas, dL_dxs, y):

        dL_dtheta = jnp.zeros_like(theta)
        prev_dL_dx = jnp.zeros_like(xs[0])
        init_val = [S-1, dL_dtheta, prev_dL_dx]

        def cond_fun(val):
            return val[0] > -1

        def body_fun(val):
            s = val[0]
            dL_dtheta, prev_dL_dx = val[1:] 
            dL_dtheta, prev_dL_dx = backwards_step(theta, dL_dtheta, us[s,:], ds[s], xs[s], 
                                                xLs[s], xRs[s], alphas[s], dL_dxs[s], prev_dL_dx, y)
            val[0] -= 1
            return [val[0], dL_dtheta, prev_dL_dx]

        val = lax.while_loop(cond_fun, body_fun, init_val)
        dL_dtheta, prev_dL_dx = val[1:]
        return dL_dtheta, prev_dL_dx

    vmapped_backwards = vmap(backwards, (None, None, 0, 0, 0, 0, 0, 0, 0, 0))

    @custom_vjp
    def slice_sample(theta, x0, ys, key):
        forwards_out = forwards_sample(theta, x0, ys, key)
        xs = forwards_out[0][:, 1:, :] # return all samples except initial condition
        return xs 

    def slice_sample_fwd(theta, x0, ys, key):
        forwards_out = forwards_sample(theta, x0, ys, key)
        xs = forwards_out[0][:, 1:, :] # return all samples except initial condition
        return xs, (forwards_out, theta, ys)

    def slice_sample_bwd(res, g):
        # g has size of xs in slice sample
        # grad theta, needs to be size of theta
        # grad_x0 , needs to be size of x0
        forwards_out, theta, ys = res
        xs0, us, ds, xLs, xRs, alphas = forwards_out
        grad_thetas, grad_x0 = vmapped_backwards(
            S, theta, us, ds, xs0, xLs, xRs, alphas, g, ys)
        grad_theta = jnp.sum(grad_thetas, axis=0)
        return (grad_theta, grad_x0, None, None)

    slice_sample.defvjp(slice_sample_fwd, slice_sample_bwd)
    slice_sample = jit(slice_sample)

    return slice_sample


# def setup_slice_sampler_with_args(log_pdf, D, S, num_chains=1):
#     """This function takes as input the log pdf, parameters.
#         It returns a differentiable slice sampling function (using custom vjp).
#         The function generates (S) samples from (num_chains) number of chains.
#         In this case, the log pdf takes a third argument.
        
#         log_pdf(x, theta, y)
#         """
    
#     def log_pdf_theta(theta, x, y):    return log_pdf(x, theta, y)
#     def log_pdf_x(x, theta, y):        return log_pdf(x, theta, y)
#     def log_pdf_ad(x, theta, a, d, y): return log_pdf(x + a * d, theta, y)
#     grad_x = jit(grad(log_pdf_x))
#     grad_theta = jit(grad(log_pdf_theta))
#     grad_x_ad = jit(grad(log_pdf_ad))

#     def forwards_step(x, theta, u1, u2, d, y):
#         func = lambda alpha : log_pdf(x + alpha * d, theta, y) - log_pdf(x, theta, y) - jnp.log(u1) # root
#         aL, bR = choose_start(func)
#         z_L, z_R = dual_bisect_method(func, aL=aL, bL=-1e-10, aR=1e-10, bR=bR)
#         x_L = x + d*z_L
#         x_R = x + d*z_R
#         x = (1 - u2) * x_L + u2 * x_R
#         alphas = jnp.array([z_L, z_R])
#         return x, x_L, x_R, alphas

#     def forwards_sample(theta, x0, ys, key):
#         # generate randomness 
#         key, *subkeys = random.split(key, 3)
#         us = random.uniform(subkeys[0], (num_chains, S, 2))
#         ds_unnorm = random.normal(subkeys[1], (S * num_chains, D))
#         ds = ds_unnorm / jnp.sqrt(jnp.sum(ds_unnorm**2, axis=1))[:,None]
#         ds = ds.reshape((num_chains, S, D))

#         xs = jnp.zeros((num_chains, S+1, D))
#         xs = index_update(xs, index[:, 0, :], x0)
#         xLs = jnp.zeros((num_chains, S, D))
#         xRs = jnp.zeros((num_chains, S, D))
#         alphas = jnp.zeros((num_chains, S, 2))
#         init_val = [xs, xLs, xRs, alphas, x0]

#         def body_fun(i, val):
#             xs, xLs, xRs, alphas, x = val 
#             x, x_L, x_R, alpha = vmap(forwards_step, (0,None,0,0,0,0))(x, theta, us[:,i,0], us[:,i,1], ds[:,i,:], ys)
#             xs = index_update(xs, index[:, i+1, :], x)
#             xLs = index_update(xLs, index[:, i, :], x_L)
#             xRs = index_update(xRs, index[:, i, :], x_R)
#             alphas = index_update(alphas, index[:, i, :], alpha)
#             val = [xs, xLs, xRs, alphas, x]
#             return val

#         xs, xLs, xRs, alphas, x = lax.fori_loop(0, S, body_fun, init_val)
#         return xs, us, ds, xLs, xRs, alphas

#     def backwards_step(theta, dL_dtheta, us, d, x, xL, xR, alphas, dL_dx, prev_dL_dx, y):

#         u1 = us[0]
#         u2 = us[1]
#         z_L = alphas[0]
#         z_R = alphas[1]

#         # compute loss for current sample
#         # set prev_dL_dx to zero at first
#         dL_dx_s = dL_dx + prev_dL_dx

#         # compute gradients of xL and xR wrt theta
#         L_grad_theta = -1.0 * (grad_theta(theta, xL, y) - grad_theta(theta, x, y)) / jnp.dot(d, grad_x_ad(x, theta, z_L, d, y))
#         R_grad_theta = -1.0 * (grad_theta(theta, xR, y) - grad_theta(theta, x, y)) / jnp.dot(d, grad_x_ad(x, theta, z_R, d, y))

#         # compute gradient dL / dtheta
#         dLd = jnp.dot(dL_dx_s, d) # dot product between loss gradient and direction - this is used multiple times 
#         dL_dtheta_s = u2 * dLd * R_grad_theta + (1-u2) * dLd * L_grad_theta
#         dL_dtheta = dL_dtheta + dL_dtheta_s

#         # propagate loss backwards : compute gradient times Jacobian of dx_s  / dx_{s-1}
#         L_grad_x = -1.0 * ( grad_x_ad(x, theta, z_L, d, y) - grad_x(x, theta, y) ) / jnp.dot(d, grad_x_ad(x, theta, z_L, d, y))
#         R_grad_x = -1.0 * ( grad_x_ad(x, theta, z_R, d, y) - grad_x(x, theta, y) ) / jnp.dot(d, grad_x_ad(x, theta, z_R, d, y))
#         prev_dL_dx = dL_dx_s + u2 * dLd * R_grad_x + (1-u2) * dLd * L_grad_x

#         return dL_dtheta, prev_dL_dx

#     def backwards(S, theta, us, ds, xs, xLs, xRs, alphas, dL_dxs, y):

#         dL_dtheta = jnp.zeros_like(theta)
#         prev_dL_dx = jnp.zeros_like(xs[0])
#         init_val = [S-1, dL_dtheta, prev_dL_dx]

#         def cond_fun(val):
#             return val[0] > -1

#         def body_fun(val):
#             s = val[0]
#             dL_dtheta, prev_dL_dx = val[1:] 
#             dL_dtheta, prev_dL_dx = backwards_step(theta, dL_dtheta, us[s,:], ds[s], xs[s], 
#                                                 xLs[s], xRs[s], alphas[s], dL_dxs[s], prev_dL_dx, y)
#             val[0] -= 1
#             return [val[0], dL_dtheta, prev_dL_dx]

#         val = lax.while_loop(cond_fun, body_fun, init_val)
#         dL_dtheta, prev_dL_dx = val[1:]
#         return dL_dtheta, prev_dL_dx

#     vmapped_backwards = vmap(backwards, (None, None, 0, 0, 0, 0, 0, 0, 0, 0))

#     @custom_vjp
#     def slice_sample(theta, x0, ys, key):
#         forwards_out = forwards_sample(theta, x0, ys, key)
#         xs = forwards_out[0][:, 1:, :] # return all samples except initial condition
#         return xs 

#     def slice_sample_fwd(theta, x0, ys, key):
#         forwards_out = forwards_sample(theta, x0, ys, key)
#         xs = forwards_out[0][:, 1:, :] # return all samples except initial condition
#         return xs, (forwards_out, theta, ys)

#     def slice_sample_bwd(res, g):
#         # g has size of xs in slice sample
#         # grad theta, needs to be size of theta
#         # grad_x0 , needs to be size of x0
#         forwards_out, theta, ys = res
#         xs0, us, ds, xLs, xRs, alphas = forwards_out
#         grad_thetas, grad_x0 = vmapped_backwards(
#             S, theta, us, ds, xs0, xLs, xRs, alphas, g, ys)
#         grad_theta = jnp.sum(grad_thetas, axis=0)
#         return (grad_theta, grad_x0, None, None)

#     slice_sample.defvjp(slice_sample_fwd, slice_sample_bwd)
#     slice_sample = jit(slice_sample)

#     return slice_sample


# if __name__ == "__main__":

#     # set up randomness
#     key = random.PRNGKey(131313)

#     # Set up params
#     D = 5   # number of dimensions
#     scale = 0.1
#     key, *subkeys = random.split(key, 3)
#     _params = [scale * random.normal(subkeys[0], (D, )), scale * random.normal(subkeys[1], (D, ))]

#     def _log_pdf(x, params):
#         mu = params[0]
#         sigma_diag = jnp.exp(params[1])
#         return jnp.sum(-0.5 * (x - mu) **2 / sigma_diag)

#     params, unflatten = ravel_pytree(_params)
#     log_pdf = jit(lambda x, params : _log_pdf(x, unflatten(params)))
#     vmapped_log_pdf = jit(vmap(log_pdf, (0,None)))

#     xstar = jnp.zeros(D)
#     Sigma = jnp.eye(D)

#     def gaussian_log_pdf(x, mu, Sigma):
#         out = -0.5 * (x - mu).T @ jnp.linalg.inv(Sigma) @ (x - mu)
#         out = out - 0.5 *  jnp.log(jnp.linalg.det(Sigma))
#         out = out - D / 2.0 * jnp.log(2.0 * jnp.pi)
#         return out

#     vmap_gaussian_log_pdf = vmap(gaussian_log_pdf, (0, None, None))

#     num_chains = 50000
#     S = 50
#     slice_sample = setup_slice_sampler(log_pdf, D, S, num_chains=num_chains)

#     from jax.lax import stop_gradient
#     def loss(params, x0, key):
#         xs_all = slice_sample(params, x0, key)
#         xs = xs_all[:, -1, :]
#         # xs = xs.reshape((S * num_chains), D)
#         loss = -1.0 * jnp.mean(vmap_gaussian_log_pdf(xs, xstar, Sigma)) 
#         loss = loss + jnp.mean(vmapped_log_pdf(xs, params)) 
#         return loss

#     grad_loss = jit(grad(loss))

#     key, *subkeys = random.split(key, 3)
#     x0 = random.normal(subkeys[0], (num_chains, D))
#     grad_params_ad = grad_loss(params, x0, subkeys[1]) 

#     def log_pdf_theta(theta, x):    return log_pdf(x, theta)
#     grad_theta = jit(grad(log_pdf_theta))
#     # grad log normalizer of posterior
#     vmapped_grad_theta = jit(vmap(grad_theta, (None,0)))

#     xs_all = slice_sample(params, x0, key)
#     xs = xs_all[:, -1, :]
#     dL_dtheta = jnp.mean(vmapped_grad_theta(params, xs), axis=0)

#     def true_loss(params):
#         mu, log_sigsqr = params
#         return 0.5 * jnp.sum(jnp.exp(log_sigsqr) + mu**2 + 1.0 - log_sigsqr)

#     true_grad = grad(lambda params : true_loss(unflatten(params)))
#     true_grad(params)

#     print(grad_params_ad - dL_dtheta)
#     print(true_grad(params))
#     # assert jnp.linalg.norm(dL_dtheta - true_grad(params)) < 1e-2
