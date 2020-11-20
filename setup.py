from setuptools import setup
import os.path as p

here = p.abspath(p.dirname(__file__))
with open(p.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='neuroswarms',
    version='1.0.1',
    description='NeuroSwarms: A neural swarming controller model',
    long_description=long_description,
    url='https://github.com/jdmonaco/neuroswarms',
    author='Joseph Monaco',
    author_email='jmonaco@jhu.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 7 - Inactive',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8'
    ],
    packages=['neuroswarms'],
)
