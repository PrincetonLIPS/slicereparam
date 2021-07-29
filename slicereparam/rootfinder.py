import jax.numpy as jnp
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
    saL, sbL, saR, sbR = jnp.sign(batch_func(jnp.array([aL, bL, aR, bR])))

    init_val = [aL, bL, aR, bR, saL, sbL, saR, sbR, i]

    def cond_fun(val):
        aL, bL, aR, bR, saL, sbL, saR, sbR, i = val
        return jnp.sum(bL-aL) + jnp.sum(bR-aR) + 100 * jnp.minimum(i, 0.0) > tol

    def body_fun(val):

        # unpack val
        aL, bL, aR, bR, saL, sbL, saR, sbR, i = val

        # new center points
        cL = (aL+bL)/2.0
        cR = (aR+bR)/2.0
        scL, scR = jnp.sign(batch_func(jnp.array([cL, cR])))

        # L
        aL = jnp.sum(cL * jnp.maximum( scL * saL, 0.0) + \
            aL * jnp.maximum( -1.0 * scL * saL, 0.0))
        bL = jnp.sum(cL * jnp.maximum( scL * sbL, 0.0) + \
            bL * jnp.maximum( -1.0 * scL * sbL, 0.0))
        saL = jnp.sum(scL * jnp.maximum( scL * saL, 0.0) + \
            saL * jnp.maximum( -1.0 * scL * saL, 0.0))
        sbL = jnp.sum(scL * jnp.maximum( scL * sbL, 0.0) + \
            sbL * jnp.maximum( -1.0 * scL * sbL, 0.0))

        # R
        aR = jnp.sum(cR * jnp.maximum( scR * saR, 0.0) + \
            aR * jnp.maximum( -1.0 * scR * saR, 0.0))
        bR = jnp.sum(cR * jnp.maximum( scR * sbR, 0.0) + \
            bR * jnp.maximum( -1.0 * scR * sbR, 0.0))
        saR = jnp.sum(scR * jnp.maximum( scR * saR, 0.0) + \
            saR * jnp.maximum( -1.0 * scR * saR, 0.0))
        sbR = jnp.sum(scR * jnp.maximum( scR * sbR, 0.0) + \
            sbR * jnp.maximum( -1.0 * scR * sbR, 0.0))

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
    log_start = -3.0, log_space = 1.0 / 5.0):

    batch_func = vmap(func, (0))

    i = 0
    aL = -1.0 * jnp.power(10.0, log_start + i * log_space)
    bR = jnp.power(10.0, log_start + i * log_space)
    aL_val, bR_val = batch_func(jnp.array([aL, bR]))
    init_val = [aL, bR, aL_val, bR_val, i]

    def cond_fun(val):
        aL, bR, aL_val, bR_val, i = val
        return jnp.maximum(aL_val, 0.0) + jnp.maximum(bR_val, 0.0) > 0.0

    def body_fun(val):

        aL, bR, aL_val, bR_val, i = val
        i = i+1
        sign_aL = jnp.sign(aL_val)
        sign_bR = jnp.sign(bR_val)
        aL = jnp.sum(-1.0 * jnp.power(10.0, log_start + i * log_space) * jnp.maximum(sign_aL, 0.0) \
                + aL * jnp.maximum(-1.0 * sign_aL, 0.0))
        bR = jnp.sum(jnp.power(10.0, log_start + i * log_space) * jnp.maximum(sign_bR, 0.0) \
                + bR * jnp.maximum(-1.0 * sign_bR, 0.0))
        aL_val, bR_val = batch_func(jnp.array([aL, bR]))
        val = [aL, bR, aL_val, bR_val, i]

        return val

    val = lax.while_loop(cond_fun, body_fun, init_val)
    aL, bR, aL_val, bR_val, i = val
    return [aL, bR]

@partial(jit, static_argnums=(0,))
def bisect_method(
    func, a=-1e5, b=-1e-5,
    tol=1e-6, maxiter=100):

    batch_func = vmap(func, (0))
    i = maxiter-1.0
    sa, sb = jnp.sign(batch_func(jnp.array([a, b])))

    init_val = [a, b, sa, sb, i]

    def cond_fun(val):
        a, b, sa, sb, i = val
        return jnp.sum(b-a) + 100 * jnp.minimum(i, 0.0) > tol

    def body_fun(val):

        # unpack val
        a, b, sa, sb, i = val

        # new center points
        c = (a+b)/2.0
        sc = jnp.sign(func(c))

        # L
        a = jnp.sum(c * jnp.maximum( sc * sa, 0.0) + \
            a * jnp.maximum( -1.0 * sc * sa, 0.0))
        b = jnp.sum(c * jnp.maximum( sc * sb, 0.0) + \
            b * jnp.maximum( -1.0 * sc * sb, 0.0))
        sa = jnp.sum(sc * jnp.maximum( sc * sa, 0.0) + \
            sa * jnp.maximum( -1.0 * sc * sa, 0.0))
        sb = jnp.sum(sc * jnp.maximum( sc * sb, 0.0) + \
            sb * jnp.maximum( -1.0 * sc * sb, 0.0))

        i = i - 1
        val = [a, b, sa, sb, i]

        return val

    val = lax.while_loop(cond_fun, body_fun, init_val)

    # unpack val
    a, b, sa, sb, i = val

    # new center points
    c = (a+b)/2.0

    return c

@partial(jit, static_argnums=(0,))
def single_choose_start(
    func,
    log_start = -3.0, log_space = 1.0 / 5.0):

    i = 0
    a = 1.0 * jnp.power(10.0, log_start + i * log_space)
    a_val = func(a)
    init_val = [a, a_val, i]

    def cond_fun(val):
        a, a_val, i = val
        return jnp.maximum(a_val, 0.0) + 100 * jnp.minimum(100. - i, 0.0) > 0.0

    def body_fun(val):

        a, a_val, i = val
        i = i+1
        sign_a = jnp.sign(a_val)
        a = jnp.sum(jnp.power(10.0, log_start + i * log_space) * jnp.maximum(sign_a, 0.0) \
                + a * jnp.maximum(-1.0 * sign_a, 0.0))
        a_val = func(a)
        val = [a, a_val, i]

        return val

    val = lax.while_loop(cond_fun, body_fun, init_val)
    a, a_val, i = val
    return a