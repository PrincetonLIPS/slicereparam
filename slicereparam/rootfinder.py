import jax.numpy as np 
from jax import jit, grad, vmap
from jax import random
from jax import lax
from jax.ops import index, index_update
from jax.flatten_util import ravel_pytree
from functools import partial

@partial(jit, static_argnums=(0,))
def dual_bisect_method(
    func,
    aL=-1e5, bL=-1e-5, aR=1e-5, bR=1e5,
    tol=1e-6, maxiter=100):

    batch_func = vmap(func, (0))
    i = maxiter-1.0
    saL, sbL, saR, sbR = np.sign(batch_func(np.array([aL, bL, aR, bR])))

    init_val = [aL, bL, aR, bR, saL, sbL, saR, sbR, i]

    def cond_fun(val):
        aL, bL, aR, bR, saL, sbL, saR, sbR, i = val
        return np.sum(bL-aL) + np.sum(bR-aR) + 100 * np.minimum(i, 0.0) > tol

    def body_fun(val):

        # unpack val
        aL, bL, aR, bR, saL, sbL, saR, sbR, i = val

        # new center points
        cL = (aL+bL)/2.0
        cR = (aR+bR)/2.0
        scL, scR = np.sign(batch_func(np.array([cL, cR])))

        # L
        aL = np.sum(cL * np.maximum( scL * saL, 0.0) + \
            aL * np.maximum( -1.0 * scL * saL, 0.0))
        bL = np.sum(cL * np.maximum( scL * sbL, 0.0) + \
            bL * np.maximum( -1.0 * scL * sbL, 0.0))
        saL = np.sum(scL * np.maximum( scL * saL, 0.0) + \
            saL * np.maximum( -1.0 * scL * saL, 0.0))
        sbL = np.sum(scL * np.maximum( scL * sbL, 0.0) + \
            sbL * np.maximum( -1.0 * scL * sbL, 0.0))

        # R
        aR = np.sum(cR * np.maximum( scR * saR, 0.0) + \
            aR * np.maximum( -1.0 * scR * saR, 0.0))
        bR = np.sum(cR * np.maximum( scR * sbR, 0.0) + \
            bR * np.maximum( -1.0 * scR * sbR, 0.0))
        saR = np.sum(scR * np.maximum( scR * saR, 0.0) + \
            saR * np.maximum( -1.0 * scR * saR, 0.0))
        sbR = np.sum(scR * np.maximum( scR * sbR, 0.0) + \
            sbR * np.maximum( -1.0 * scR * sbR, 0.0))

        i = i - 1
        val = [aL, bL, aR, bR, saL, sbL, saR, sbR, i]

        return val

    val = lax.while_loop(cond_fun, body_fun, init_val)

    # unpack val
    aL, bL, aR, bR, saL, sbL, saR, sbR, i = val

    # new center points
    cL = (aL+bL)/2.0
    cR = (aR+bR)/2.0

    return [cL, cR]

@partial(jit, static_argnums=(0,))
def choose_start(
    func,
    log_start = -2.0, log_space = 1.0 / 3.0):

    batch_func = vmap(func, (0))

    i = 0
    aL = -1.0 * np.power(10.0, log_start + i * log_space)
    bR = np.power(10.0, log_start + i * log_space)
    aL_val, bR_val = batch_func(np.array([aL, bR]))
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
        aL_val, bR_val = batch_func(np.array([aL, bR]))
        val = [aL, bR, aL_val, bR_val, i]

        return val

    val = lax.while_loop(cond_fun, body_fun, init_val)
    aL, bR, aL_val, bR_val, i = val
    return [aL, bR]