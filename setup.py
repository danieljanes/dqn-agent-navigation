from setuptools import setup
from setuptools import find_packages


setup(
    name="dqn-navigation",
    version="0.0.1",
    license="Proprietary",
    packages=find_packages('src'),
    package_dir={'': 'src'},
)
