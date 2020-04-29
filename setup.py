#!/usr/bin/env python

from distutils.core import setup

setup(name='slicereparam',
      version='0.1.0',
      description='slice sampling reparameterization',
      authors='David Zoltowski',
      url='',
      packages=['slicereparam'],
      install_requires=['numpy','scipy','matplotlib','jax','jaxlib']
      )
