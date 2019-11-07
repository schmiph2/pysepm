# pysepm - Python Speech Enhancement Performance Measures (Quality and Intelligibility)
[![DOI](https://zenodo.org/badge/220233987.svg)](https://zenodo.org/badge/latestdoi/220233987)

Python implementation of objective quality and intelligibilty measures mentioned in Philipos C. Loizou's great [Speech Enhancement Book](https://www.crcpress.com/Speech-Enhancement-Theory-and-Practice-Second-Edition/Loizou/p/book/9781138075573). The Python implementations are checked with the MATLAB implementations attached to the book (see [Link](https://crcpress.com/downloads/K14513/K14513_CD_Files.zip))

# Install with pip
Install pysepm:
```
pip3 install https://github.com/schmiph2/pysepm/archive/master.zip
```
# Examples
Please find a Jupyter Notebook with examples for all implemented measures in the [examples folder](https://github.com/schmiph2/pysepm/tree/master/examples).

# Implemented Measures
## Speech Quality Measures
+ Segmental Signal-to-Noise Ratio (SNRseg)
+ Frequency-weighted Segmental SNR (fwSNRseg)
+ Log-likelihood Ratio (LLR)
+ Weighted Spectral Slope (WSS)
+ Perceptual Evaluation of Speech Quality (PESQ), ([python-pesq](https://github.com/ludlows/python-pesq) implementation by ludlows)
+ Composite Objective Speech Quality (composite)
+ Cepstrum Distance Objective Speech Quality Measure (CD)

## Speech Intelligibility Measures
+ Short-time objective intelligibility (STOI), ([pystoi](https://github.com/mpariente/pystoi) implementation by mpariente)
+ Coherence and speech intelligibility index (CSII)
+ Normalized-covariance measure (NCM)

## Dereverberation Measures (TODO)
+ Bark spectral distortion (BSD) 
+ Scale-invariant signal to distortion ratio (SI-SDR)
