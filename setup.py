from setuptools import setup
from setuptools import find_packages

setup(
    name='pysepm',
    version='0.1',
    description='Computes Objective Quality measures',
    author='Philipp Schmid',
    author_email='scdp@zhaw.ch',
    url='https://github.zhaw.ch/scdp/py-sepm',
    license='MIT',
    install_requires=[
	    	'cython',
	    	'wheel',
	    	'numpy',
		'scipy',
		'pystoi',
		'pesq',
	],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages()
)
