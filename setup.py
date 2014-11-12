from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name = "starutils",
      version = "0.1",
      description = "Useful things for playing with simulated star populations.",
      long_description = readme(),
      author = "Timothy D. Morton",
      author_email = "tim.morton@gmail.com",
      url = "https://github.com/timothydmorton/starutils",
      packages = find_packages(),
      package_data = {'starutils': ['data/*']},
      scripts = ['scripts/get_trilegal'],
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Astronomy'
        ],
      install_requires=['pandas>=0.13','plotutils','orbitutils'],
      zip_safe=False
) 
