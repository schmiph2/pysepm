# pysepm - Speech Enhancement Performance Measures (Quality and Intelligibility) Implemented in Python
Python implementation of objective quality and intelligibilty measures mentioned in Philipos C. Loizou's great [Speech Enhancement Book](https://www.crcpress.com/Speech-Enhancement-Theory-and-Practice-Second-Edition/Loizou/p/book/9781138075573). The Python implementations are checked with the MATLAB implementations attached to the book (see [Link](https://crcpress.com/downloads/K14513/K14513_CD_Files.zip))


# Requirements

    gcc compiler
    cython
    numpy
    scipy
    pystoi
    pypesq 
    
# Implemented Speech Quality Measures
+ Segmental Signal-to-Noise Ratio (SNRseg)
+ Frequency-weighted Segmental SNR (fwSNRseg)
+ Log-likelihood Ratio (llr)
+ Weighted Spectral Slope (wss)
+ Perceptual Evaluation of Speech Quality (pesq)
+ Composite Objective Speech Quality (composite)

# Implemented Speech Intelligibility Measures
+ Short-time objective intelligibility (stoi)
+ Coherence and speech intelligibility index (CSII)
+ Normalized-covariance measure (NCM)

# Install with pip
The setup.py is not finished yet. You have to install the dependencies by hand.

First install Cython (Required for python-pesq):
```
pip3 install Cython
```
Then install python-pesq and pystoi:
```
pip3 install https://github.com/schmiph2/python-pesq/archive/master.zip
pip3 install https://github.com/schmiph2/pystoi/archive/master.zip
```
Finally install pysepm:
```
pip3 install https://github.com/schmiph2/pysepm/archive/master.zip
```


# TODO:
+ Finish setup script
+ systematic tests for all measures
+ change resample method used (e.g. in NCM) to fit matlab implementation
