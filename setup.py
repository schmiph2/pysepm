from setuptools import setup,find_packages

setup(
    name='pysepm',
    version='0.1',
    description='Computes Objective Quality measures',
    author='Philipp Schmid',
    author_email='scdp@zhaw.ch',
    url='https://github.zhaw.ch/scdp/pysepm',
    license='MIT',
    install_requires=[
	    'numpy',
		'scipy',
        'numba',
		'pystoi',
		'pesq @ https://github.com/ludlows/python-pesq/archive/master.zip#egg=pesq',
		'SRMRpy @  https://github.com/jfsantos/SRMRpy/archive/master.zip#egg=SRMRpy',
	],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages()
)
