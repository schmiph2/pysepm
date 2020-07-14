# pysepm - Speech Enhancement Performance Measures (Quality and Intelligibility) Implemented in Python
Python implementation of objective quality and intelligibilty measures mentioned in Philipos C. Loizou's great [Speech Enhancement Book](https://www.crcpress.com/Speech-Enhancement-Theory-and-Practice-Second-Edition/Loizou/p/book/9781138075573). The Python implementations are checked with the MATLAB implementations attached to the book (see [Link](https://crcpress.com/downloads/K14513/K14513_CD_Files.zip))

# Contribution
If you would like to have an additional measure or if you find a bug/expected behaviour please create an issue.

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
+ Perceptual Evaluation of Speech Quality (PESQ)
+ Composite Objective Speech Quality (composite)
+ Cepstrum Distance Objective Speech Quality Measure (CD)

## Speech Intelligibility Measures
+ Short-time objective intelligibility (STOI)
+ Coherence and speech intelligibility index (CSII)
+ Normalized-covariance measure (NCM)

## Dereverberation Measures (TODO)
+ Bark spectral distortion (BSD) 
+ Scale-invariant signal to distortion ratio (SI-SDR)
