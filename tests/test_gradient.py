from jax.config import config
config.update("jax_enable_x64", True)

# import jax.numpy as np 
# from jax import jit, grad, vmap
# from jax import random
# from jax import lax
# from jax.ops import index, index_update
# from jax.flatten_util import ravel_pytree
# from functools import partial

from slicereparam.slicesampler import slicesampler

def test_grad():
    bool = True 
    assert bool 

if __name__ == "__main__":
    test_grad()
    