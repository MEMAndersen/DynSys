# -*- coding: utf-8 -*-
"""
Setup file for `dynsys` package
"""

# Always prefer setuptools over distutils
from setuptools import setup
from os import path

import dynsys

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(name='DynSys',
      version=dynsys.__version__,
      description='Functions and classes to faciliate dynamic analysis',
      long_description=long_description,
      url='<Insert portal link here>',
      author=dynsys.__author__,
      author_email='rihy@cowi.com',  # Optional
      version=dynsys.__version__,
      
      classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Structural analysis :: Dynamic analysis',

        # Pick your license as you wish
        #'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
      ],

      keywords='dynamics time-stepping moving-load',
      packages=["dynsys"]
     )

