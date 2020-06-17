from scipy.signal import resample
import srmrpy #https://github.com/jfsantos/SRMRpy

from .qualityMeasures import SNRseg

def srr_seg(clean_speech, processed_speech,fs):
    return SNRseg(clean_speech, processed_speech,fs)


def srmr(speech,fs, n_cochlear_filters=23, low_freq=125, min_cf=4, max_cf=128, fast=True, norm=False):    
    if fs == 8000:
        return srmr(speech, fs, n_cochlear_filters=n_cochlear_filters, low_freq=low_freq, min_cf=min_cf, max_cf=max_cf, fast=fast, norm=norm)

    elif fs == 16000:
        return srmr(speech, fs, n_cochlear_filters=n_cochlear_filters, low_freq=low_freq, min_cf=min_cf, max_cf=max_cf, fast=fast, norm=norm)
    
    else:
        numSamples=round(len(speech)/fs*16000)
        fs = 16000
        return srmr(resample(speech, numSamples), fs, n_cochlear_filters=n_cochlear_filters, low_freq=low_freq, min_cf=min_cf, max_cf=max_cf, fast=fast, norm=norm)


    
def rdt(clean_speech, processed_speech,fs):
    pass


def bsd(clean_speech, processed_speech,fs):
    pass


        
        
    
