from setuptools import setup

setup(
    name='bo',
    version='0.1.0',    
    description='Bayesian Optimisation',
    packages=['src'],
    install_requires=['mpi4py>=2.0',
                      'numpy',                     
                      ],
)
