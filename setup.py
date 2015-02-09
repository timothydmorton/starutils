from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import os, sys

def readme():
    with open('README.rst') as f:
        return f.read()

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
import sys
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__STARUTILS_SETUP__ = True
import starutils
version = starutils.__version__


# Publish the library to PyPI.
if "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload")
    sys.exit()

# Push a new tag to GitHub.
if "tag" in sys.argv:
    os.system("git tag -a {0} -m 'version {0}'".format(version))
    os.system("git push --tags")
    sys.exit()


setup(name = "starutils",
      version = version,
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
      install_requires=['pandas>=0.13','plotutils','orbitutils',
                        'simpledist','isochrones','hashutils'],
      zip_safe=False
) 
