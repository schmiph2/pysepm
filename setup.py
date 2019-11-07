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
    install_requires=['numpy',
		 'scipy',
		'pystoi @ https://github.com/schmiph2/pystoi/archive/master.zip#egg=pystoi',
		'pypesq',
	],
	dependency_links=[
		'git://git@github.com/schmiph2/python-pesq.git#egg=pypesq',
	]
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages()
)
