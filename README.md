# pysepm - Speech Enhancement Performance Measures (Quality and Intelligibility) Implemented in Python
Python implementation of objective quality and intelligibilty measures mentioned in Philipos C. Loizou's great [Speech Enhancement Book](https://www.crcpress.com/Speech-Enhancement-Theory-and-Practice-Second-Edition/Loizou/p/book/9781138075573). The Python implementations are checked with the MATLAB implementations attached to the book.


# Requirements

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
