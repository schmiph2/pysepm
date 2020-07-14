from scipy.signal import resample,stft
import scipy
import srmrpy #https://github.com/jfsantos/SRMRpy
import numpy as np
from .qualityMeasures import SNRseg

def srr_seg(clean_speech, processed_speech,fs):
    return SNRseg(clean_speech, processed_speech,fs)


def srmr(speech,fs, n_cochlear_filters=23, low_freq=125, min_cf=4, max_cf=128, fast=False, norm=False):    
    if fs == 8000:
        srmRatio,energy=srmrpy.srmr(speech, fs, n_cochlear_filters=n_cochlear_filters, low_freq=low_freq, min_cf=min_cf, max_cf=max_cf, fast=fast, norm=norm)
        return srmRatio

    elif fs == 16000:
        srmRatio,energy=srmrpy.srmr(speech, fs, n_cochlear_filters=n_cochlear_filters, low_freq=low_freq, min_cf=min_cf, max_cf=max_cf, fast=fast, norm=norm)
        return srmRatio
    else:
        numSamples=round(len(speech)/fs*16000)
        fs = 16000
        srmRatio,energy=srmrpy.srmr(resample(speech, numSamples), fs, n_cochlear_filters=n_cochlear_filters, low_freq=low_freq, min_cf=min_cf, max_cf=max_cf, fast=fast, norm=norm)
        return srmRatio 


def hz_to_bark(freqs_hz):
    freqs_hz = np.asanyarray([freqs_hz])
    barks = (26.81*freqs_hz)/(1960+freqs_hz)-0.53
    barks[barks<2]=barks[barks<2]+0.15*(2-barks[barks<2])
    barks[barks>20.1]=barks[barks>20.1]+0.22*(barks[barks>20.1]-20.1)
    return np.squeeze(barks)

def bark_to_hz(barks):
    barks = barks.copy()
    barks = np.asanyarray([barks])
    barks[barks<2]=(barks[barks<2]-0.3)/0.85
    barks[barks>20.1]=(barks[barks>20.1]+4.422)/1.22
    freqs_hz = 1960 * (barks+0.53)/(26.28-barks)
    return np.squeeze(freqs_hz)

def bark_frequencies(n_barks=128, fmin=0.0, fmax=11025.0):
    # 'Center freqs' of bark bands - uniformly spaced between limits
    min_bark = hz_to_bark(fmin)
    max_bark = hz_to_bark(fmax)

    barks = np.linspace(min_bark, max_bark, n_barks)

    return bark_to_hz(barks)

def barks(fs, n_fft, n_barks=128, fmin=0.0, fmax=None, norm='area', dtype=np.float32):

    if fmax is None:
        fmax = float(fs) / 2


    # Initialize the weights
    n_barks = int(n_barks)
    weights = np.zeros((n_barks, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = np.linspace(0,float(fs) / 2,int(1 + n_fft//2), endpoint=True)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    bark_f = bark_frequencies(n_barks + 2, fmin=fmin, fmax=fmax)

    fdiff = np.diff(bark_f)
    ramps = np.subtract.outer(bark_f, fftfreqs)

    for i in range(n_barks):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))        

    if norm in (1, 'area'):
        weightsPerBand=np.sum(weights,1);
        for i in range(weights.shape[0]):
            weights[i,:]=weights[i,:]/weightsPerBand[i]
    return weights

def bsd(clean_speech, processed_speech, fs, frameLen=0.03, overlap=0.75):
    
    pre_emphasis_coeff = 0.95
    b = np.array([1])
    a = np.array([1,pre_emphasis_coeff])
    clean_speech = scipy.signal.lfilter(b,a,clean_speech)
    processed_speech = scipy.signal.lfilter(b,a,processed_speech)

    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples
    max_freq    = fs/2 #maximum bandwidth
    n_fft       = 2**np.ceil(np.log2(2*winlength))
    n_fftby2    = int(n_fft/2)
    num_frames = len(clean_speech)/skiprate-(winlength/skiprate)# number of frames
    
    hannWin=scipy.signal.windows.hann(winlength)#0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    f,t,Zxx=stft(clean_speech[0:int(num_frames)*skiprate+int(winlength-skiprate)], fs=fs, window=hannWin, nperseg=winlength, noverlap=winlength-skiprate, nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=False)
    clean_power_spec=np.square(np.sum(hannWin)*np.abs(Zxx))
    f,t,Zxx=stft(processed_speech[0:int(num_frames)*skiprate+int(winlength-skiprate)], fs=fs, window=hannWin, nperseg=winlength, noverlap=winlength-skiprate, nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=False)
    enh_power_spec=np.square(np.sum(hannWin)*np.abs(Zxx))

    bark_filt = barks(fs, n_fft, n_barks=32)
    clean_power_spec_bark= np.dot(bark_filt,clean_power_spec)
    enh_power_spec_bark= np.dot(bark_filt,enh_power_spec)
    
    clean_power_spec_bark_2=np.square(clean_power_spec_bark)
    diff_power_spec_2 = np.square(clean_power_spec_bark-enh_power_spec_bark)
    
    bsd = np.mean(np.sum(diff_power_spec_2,axis=0)/np.sum(clean_power_spec_bark_2,axis=0))
    return bsd


        
        
    
