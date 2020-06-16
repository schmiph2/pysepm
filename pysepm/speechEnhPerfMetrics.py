from scipy.signal import stft,get_window,correlate,resample,butter,lfilter,hilbert
from scipy.linalg import solve_toeplitz,toeplitz
from scipy.interpolate import interp1d
from pypesq import pypesq # https://github.com/ludlows/python-pesq
import pystoi.stoi # https://github.com/mpariente/pystoi
import numpy as np


stoi = pystoi.stoi.stoi

def extractOverlappedWindows(x,nperseg,noverlap,window=None):
    # source: https://github.com/scipy/scipy/blob/v1.2.1/scipy/signal/spectral.py
    step = nperseg - noverlap
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    if window is not None:
        result = window * result
    return result

def SNRseg(clean_speech, processed_speech,fs, frameLen=0.03, overlap=0.75):
    eps=np.finfo(np.float64).eps

    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples
    MIN_SNR     = -10 # minimum SNR in dB
    MAX_SNR     =  35 # maximum SNR in dB

    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    clean_speech_framed=extractOverlappedWindows(clean_speech,winlength,winlength-skiprate,hannWin)
    processed_speech_framed=extractOverlappedWindows(processed_speech,winlength,winlength-skiprate,hannWin)
    
    signal_energy = np.power(clean_speech_framed,2).sum(-1)
    noise_energy = np.power(clean_speech_framed-processed_speech_framed,2).sum(-1)
    
    segmental_snr = 10*np.log10(signal_energy/(noise_energy+eps)+eps)
    segmental_snr[segmental_snr<MIN_SNR]=MIN_SNR
    segmental_snr[segmental_snr>MAX_SNR]=MAX_SNR
    segmental_snr=segmental_snr[:-1] # remove last frame -> not valid
    return np.mean(segmental_snr)
	
def fwSNRseg(cleanSig, enhancedSig, fs, frameLen=0.03, overlap=0.75):
    if cleanSig.shape!=enhancedSig.shape:
        raise ValueError('The two signals do not match!')
    eps=np.finfo(np.float64).eps
    cleanSig=cleanSig.astype(np.float64)+eps
    enhancedSig=enhancedSig.astype(np.float64)+eps
    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples
    max_freq    = fs/2 #maximum bandwidth
    num_crit    = 25# number of critical bands
    n_fft       = 2**np.ceil(np.log2(2*winlength))
    n_fftby2    = int(n_fft/2)
    gamma=0.2

    cent_freq=np.zeros((num_crit,))
    bandwidth=np.zeros((num_crit,))

    cent_freq[0]  = 50.0000;   bandwidth[0]  = 70.0000;
    cent_freq[1]  = 120.000;   bandwidth[1]  = 70.0000;
    cent_freq[2]  = 190.000;   bandwidth[2]  = 70.0000;
    cent_freq[3]  = 260.000;   bandwidth[3]  = 70.0000;
    cent_freq[4]  = 330.000;   bandwidth[4]  = 70.0000;
    cent_freq[5]  = 400.000;   bandwidth[5]  = 70.0000;
    cent_freq[6]  = 470.000;   bandwidth[6]  = 70.0000;
    cent_freq[7]  = 540.000;   bandwidth[7]  = 77.3724;
    cent_freq[8]  = 617.372;   bandwidth[8]  = 86.0056;
    cent_freq[9] = 703.378;    bandwidth[9] = 95.3398;
    cent_freq[10] = 798.717;   bandwidth[10] = 105.411;
    cent_freq[11] = 904.128;   bandwidth[11] = 116.256;
    cent_freq[12] = 1020.38;   bandwidth[12] = 127.914;
    cent_freq[13] = 1148.30;   bandwidth[13] = 140.423;
    cent_freq[14] = 1288.72;   bandwidth[14] = 153.823;
    cent_freq[15] = 1442.54;   bandwidth[15] = 168.154;
    cent_freq[16] = 1610.70;   bandwidth[16] = 183.457;
    cent_freq[17] = 1794.16;   bandwidth[17] = 199.776;
    cent_freq[18] = 1993.93;   bandwidth[18] = 217.153;
    cent_freq[19] = 2211.08;   bandwidth[19] = 235.631;
    cent_freq[20] = 2446.71;   bandwidth[20] = 255.255;
    cent_freq[21] = 2701.97;   bandwidth[21] = 276.072;
    cent_freq[22] = 2978.04;   bandwidth[22] = 298.126;
    cent_freq[23] = 3276.17;   bandwidth[23] = 321.465;
    cent_freq[24] = 3597.63;   bandwidth[24] = 346.136;


    W=np.array([0.003,0.003,0.003,0.007,0.010,0.016,0.016,0.017,0.017,0.022,0.027,0.028,0.030,0.032,0.034,0.035,0.037,0.036,0.036,0.033,0.030,0.029,0.027,0.026,
    0.026])

    bw_min=bandwidth[0]
    min_factor = np.exp (-30.0 / (2.0 * 2.303));#      % -30 dB point of filter

    all_f0=np.zeros((num_crit,))
    crit_filter=np.zeros((num_crit,int(n_fftby2)))
    j = np.arange(0,n_fftby2)


    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        all_f0[i] = np.floor(f0);
        bw = (bandwidth[i] / max_freq) * (n_fftby2);
        norm_factor = np.log(bw_min) - np.log(bandwidth[i]);
        crit_filter[i,:] = np.exp (-11 *(((j - np.floor(f0))/bw)**2) + norm_factor)
        crit_filter[i,:] = crit_filter[i,:]*(crit_filter[i,:] > min_factor)

    num_frames = len(cleanSig)/skiprate-(winlength/skiprate)# number of frames
    start      = 1 # starting sample
    #window     = 0.5*(1 - cos(2*pi*(1:winlength).T/(winlength+1)));


    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    f,t,Zxx=stft(cleanSig[0:int(num_frames)*skiprate+int(winlength-skiprate)], fs=fs, window=hannWin, nperseg=winlength, noverlap=winlength-skiprate, nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=False)
    clean_spec=np.abs(Zxx)
    clean_spec=clean_spec[:-1,:]
    clean_spec=(clean_spec/clean_spec.sum(0))
    f,t,Zxx=stft(enhancedSig[0:int(num_frames)*skiprate+int(winlength-skiprate)], fs=fs, window=hannWin, nperseg=winlength, noverlap=winlength-skiprate, nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=False)
    enh_spec=np.abs(Zxx)
    enh_spec=enh_spec[:-1,:]
    enh_spec=(enh_spec/enh_spec.sum(0))

    clean_energy=(crit_filter.dot(clean_spec))
    processed_energy=(crit_filter.dot(enh_spec))
    error_energy=np.power(clean_energy-processed_energy,2)
    error_energy[error_energy<eps]=eps
    W_freq=np.power(clean_energy,gamma)
    SNRlog=10*np.log10((clean_energy**2)/error_energy)
    fwSNR=np.sum(W_freq*SNRlog,0)/np.sum(W_freq,0)
    distortion=fwSNR.copy()
    distortion[distortion<-10]=-10
    distortion[distortion>35]=35

    return np.mean(distortion)
	
	
def lpcoeff(speech_frame, model_order):
   # ----------------------------------------------------------
   # (1) Compute Autocorrelation Lags
   # ----------------------------------------------------------

    R=correlate(speech_frame,speech_frame) 
    R=R[len(speech_frame)-1:len(speech_frame)+model_order]
   # ----------------------------------------------------------
   # (2) Levinson-Durbin
   # ----------------------------------------------------------
    lpparams=np.ones((model_order+1))
    lpparams[1:]=solve_toeplitz(R[0:-1],-R[1:])
    
    
    return(lpparams,R)

def llr(clean_speech, processed_speech, fs, frameLen=0.03, overlap=0.75):
    eps=np.finfo(np.float64).eps
    alpha = 0.95
    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples    
    if fs<10000:
        P = 10 # LPC Analysis Order
    else:
        P = 16 # this could vary depending on sampling frequency.
        
    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    clean_speech_framed=extractOverlappedWindows(clean_speech,winlength,winlength-skiprate,hannWin)
    processed_speech_framed=extractOverlappedWindows(processed_speech,winlength,winlength-skiprate,hannWin)
    numFrames=clean_speech_framed.shape[0]
    numerators = np.zeros((numFrames-1,))
    denominators = np.zeros((numFrames-1,))
    
    for ii in range(numFrames-1):
        A_clean,R_clean=lpcoeff(clean_speech_framed[ii,:],P)
        A_proc,R_proc=lpcoeff(processed_speech_framed[ii,:],P)
        
        numerators[ii]=A_proc.dot(toeplitz(R_clean).dot(A_proc.T))
        denominators[ii]=A_clean.dot(toeplitz(R_clean).dot(A_clean.T))
    
    
    frac=numerators/denominators
    frac[frac<=0]=1000
    distortion = np.log(frac)
    #distortion[distortion>2]=2 # this line is not in composite measure matlab implementation of loizou
    distortion = np.sort(distortion)
    distortion = distortion[:int(round(len(distortion)*alpha))]
    return np.mean(distortion)


def findLocPeaks(slope,energy):
    num_crit = len(energy)
    
    loc_peaks=np.zeros_like(slope)

    for ii in range(len(slope)):
        n=ii
        if slope[ii]>0:
            while ((n<num_crit-1) and (slope[n] > 0)):
                n=n+1
            loc_peaks[ii]=energy[n-1]
        else:
            while ((n>=0) and (slope[n] <= 0)):
                n=n-1
            loc_peaks[ii]=energy[n+1]
            
    return loc_peaks



def wss(clean_speech, processed_speech, fs, frameLen=0.03, overlap=0.75):    
    
    Kmax        = 20 # value suggested by Klatt, pg 1280
    Klocmax     = 1 # value suggested by Klatt, pg 1280
    alpha = 0.95
    if clean_speech.shape!=processed_speech.shape:
        raise ValueError('The two signals do not match!')
    eps=np.finfo(np.float64).eps
    clean_speech=clean_speech.astype(np.float64)+eps
    processed_speech=processed_speech.astype(np.float64)+eps
    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples
    max_freq    = fs/2 #maximum bandwidth
    num_crit    = 25# number of critical bands
    n_fft       = 2**np.ceil(np.log2(2*winlength))
    n_fftby2    = int(n_fft/2)

    cent_freq=np.zeros((num_crit,))
    bandwidth=np.zeros((num_crit,))

    cent_freq[0]  = 50.0000;   bandwidth[0]  = 70.0000;
    cent_freq[1]  = 120.000;   bandwidth[1]  = 70.0000;
    cent_freq[2]  = 190.000;   bandwidth[2]  = 70.0000;
    cent_freq[3]  = 260.000;   bandwidth[3]  = 70.0000;
    cent_freq[4]  = 330.000;   bandwidth[4]  = 70.0000;
    cent_freq[5]  = 400.000;   bandwidth[5]  = 70.0000;
    cent_freq[6]  = 470.000;   bandwidth[6]  = 70.0000;
    cent_freq[7]  = 540.000;   bandwidth[7]  = 77.3724;
    cent_freq[8]  = 617.372;   bandwidth[8]  = 86.0056;
    cent_freq[9] = 703.378;    bandwidth[9] = 95.3398;
    cent_freq[10] = 798.717;   bandwidth[10] = 105.411;
    cent_freq[11] = 904.128;   bandwidth[11] = 116.256;
    cent_freq[12] = 1020.38;   bandwidth[12] = 127.914;
    cent_freq[13] = 1148.30;   bandwidth[13] = 140.423;
    cent_freq[14] = 1288.72;   bandwidth[14] = 153.823;
    cent_freq[15] = 1442.54;   bandwidth[15] = 168.154;
    cent_freq[16] = 1610.70;   bandwidth[16] = 183.457;
    cent_freq[17] = 1794.16;   bandwidth[17] = 199.776;
    cent_freq[18] = 1993.93;   bandwidth[18] = 217.153;
    cent_freq[19] = 2211.08;   bandwidth[19] = 235.631;
    cent_freq[20] = 2446.71;   bandwidth[20] = 255.255;
    cent_freq[21] = 2701.97;   bandwidth[21] = 276.072;
    cent_freq[22] = 2978.04;   bandwidth[22] = 298.126;
    cent_freq[23] = 3276.17;   bandwidth[23] = 321.465;
    cent_freq[24] = 3597.63;   bandwidth[24] = 346.136;


    W=np.array([0.003,0.003,0.003,0.007,0.010,0.016,0.016,0.017,0.017,0.022,0.027,0.028,0.030,0.032,0.034,0.035,0.037,0.036,0.036,0.033,0.030,0.029,0.027,0.026,
    0.026])

    bw_min=bandwidth[0]
    min_factor = np.exp (-30.0 / (2.0 * 2.303));#      % -30 dB point of filter

    all_f0=np.zeros((num_crit,))
    crit_filter=np.zeros((num_crit,int(n_fftby2)))
    j = np.arange(0,n_fftby2)


    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        all_f0[i] = np.floor(f0);
        bw = (bandwidth[i] / max_freq) * (n_fftby2);
        norm_factor = np.log(bw_min) - np.log(bandwidth[i]);
        crit_filter[i,:] = np.exp (-11 *(((j - np.floor(f0))/bw)**2) + norm_factor)
        crit_filter[i,:] = crit_filter[i,:]*(crit_filter[i,:] > min_factor)

    num_frames = len(clean_speech)/skiprate-(winlength/skiprate)# number of frames
    start      = 1 # starting sample

    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    scale = np.sqrt(1.0 / hannWin.sum()**2)

    f,t,Zxx=stft(clean_speech[0:int(num_frames)*skiprate+int(winlength-skiprate)], fs=fs, window=hannWin, nperseg=winlength, noverlap=winlength-skiprate, nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=False)
    clean_spec=np.power(np.abs(Zxx)/scale,2)
    clean_spec=clean_spec[:-1,:]
    
    f,t,Zxx=stft(processed_speech[0:int(num_frames)*skiprate+int(winlength-skiprate)], fs=fs, window=hannWin, nperseg=winlength, noverlap=winlength-skiprate, nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=False)
    proc_spec=np.power(np.abs(Zxx)/scale,2)
    proc_spec=proc_spec[:-1,:]

    clean_energy=(crit_filter.dot(clean_spec))
    log_clean_energy=10*np.log10(clean_energy)
    log_clean_energy[log_clean_energy<-100]=-100
    proc_energy=(crit_filter.dot(proc_spec))
    log_proc_energy=10*np.log10(proc_energy)
    log_proc_energy[log_proc_energy<-100]=-100

    log_clean_energy_slope=np.diff(log_clean_energy,axis=0)
    log_proc_energy_slope=np.diff(log_proc_energy,axis=0)

    dBMax_clean     = np.max(log_clean_energy,axis=0)
    dBMax_processed = np.max(log_proc_energy,axis=0)
    
    numFrames=log_clean_energy_slope.shape[-1]
    
    clean_loc_peaks=np.zeros_like(log_clean_energy_slope)
    proc_loc_peaks=np.zeros_like(log_proc_energy_slope)
    for ii in range(numFrames):
        clean_loc_peaks[:,ii]=findLocPeaks(log_clean_energy_slope[:,ii],log_clean_energy[:,ii])
        proc_loc_peaks[:,ii]=findLocPeaks(log_proc_energy_slope[:,ii],log_proc_energy[:,ii])
    

    Wmax_clean = Kmax / (Kmax + dBMax_clean - log_clean_energy[:-1,:])   
    Wlocmax_clean  = Klocmax / ( Klocmax + clean_loc_peaks - log_clean_energy[:-1,:])
    W_clean           = Wmax_clean * Wlocmax_clean

    Wmax_proc = Kmax / (Kmax + dBMax_processed - log_proc_energy[:-1])   
    Wlocmax_proc  = Klocmax / ( Klocmax + proc_loc_peaks - log_proc_energy[:-1,:])
    W_proc           = Wmax_proc * Wlocmax_proc

    W = (W_clean + W_proc)/2.0

    distortion=np.sum(W*(log_clean_energy_slope- log_proc_energy_slope)**2,axis=0)
    distortion=distortion/np.sum(W,axis=0)
    distortion = np.sort(distortion)
    distortion = distortion[:int(round(len(distortion)*alpha))]
    return np.mean(distortion)

def pesq(clean_speech, processed_speech, fs):
    if fs == 8000:
        pesq_mos = pypesq(fs,clean_speech, processed_speech, 'nb')
        pesq_mos = 46607/14945 - (2000*np.log(1/(pesq_mos/4 - 999/4000) - 1))/2989 #remap to raw pesq score

    elif fs == 16000:
        pesq_mos = pypesq(fs,clean_speech, processed_speech, 'wb')
    elif fs >= 16000:
        numSamples=round(len(clean_speech)/fs*16000)
        pesq_mos = pypesq(fs,resample(clean_speech, numSamples), resample(processed_speech, numSamples), 'wb')
    else:
        numSamples=round(len(clean_speech)/fs*8000)
        pesq_mos = pypesq(fs,resample(clean_speech, numSamples), resample(processed_speech, numSamples), 'nb')
        pesq_mos = 46607/14945 - (2000*np.log(1/(pesq_mos/4 - 999/4000) - 1))/2989 #remap to raw pesq score

    return pesq_mos



def composite(clean_speech, processed_speech, fs):
    wss_dist=wss(clean_speech, processed_speech, fs)
    llr_mean=llr(clean_speech, processed_speech, fs)
    segSNR=SNRseg(clean_speech, processed_speech, fs)
    pesq_mos = pesq(clean_speech, processed_speech,fs)

    Csig = 3.093 - 1.029*llr_mean + 0.603*pesq_mos-0.009*wss_dist
    Csig = np.max((1,Csig))  
    Csig = np.min((5, Csig)) # limit values to [1, 5]
    Cbak = 1.634 + 0.478 *pesq_mos - 0.007*wss_dist + 0.063*segSNR
    Cbak = np.max((1, Cbak))
    Cbak = np.min((5,Cbak)) # limit values to [1, 5]
    Covl = 1.594 + 0.805*pesq_mos - 0.512*llr_mean - 0.007*wss_dist
    Covl = np.max((1, Covl))
    Covl = np.min((5, Covl)) # limit values to [1, 5]
    return Csig,Cbak,Covl	



def fwseg_noise(clean_speech, processed_speech,sample_rate,frameLen=0.03, overlap=0.75):
    
    clean_length = len(clean_speech)
    processed_length = len(processed_speech)
    rms_all=np.linalg.norm(clean_speech)/np.sqrt(processed_length)
    
    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples
    max_freq    = fs/2 #maximum bandwidth
    num_crit    = 16 # number of critical bands
    n_fft       = int(2**np.ceil(np.log2(2*winlength)))
    n_fftby2    = int(n_fft/2)

    cent_freq=np.zeros((num_crit,))
    bandwidth=np.zeros((num_crit,))

    # ----------------------------------------------------------------------
    # Critical Band Filter Definitions (Center Frequency and Bandwidths in Hz)
    # ----------------------------------------------------------------------
    cent_freq[0]  = 150.0000;   bandwidth[0]   = 100.0000;
    cent_freq[1]  = 250.000;    bandwidth[1]  = 100.0000;
    cent_freq[2]  = 350.000;    bandwidth[2]  = 100.0000;
    cent_freq[3]  = 450.000;    bandwidth[3]  = 110.0000;
    cent_freq[4]  = 570.000;    bandwidth[4]  = 120.0000;
    cent_freq[5]  = 700.000;    bandwidth[5]  = 140.0000;
    cent_freq[6]  = 840.000;    bandwidth[6]  = 150.0000;
    cent_freq[7]  = 1000.000;   bandwidth[7]  = 160.000;
    cent_freq[8]  = 1170.000;   bandwidth[8]  = 190.000;
    cent_freq[9] = 1370.000;    bandwidth[9] = 210.000;
    cent_freq[10] = 1600.000;   bandwidth[10]= 240.000;
    cent_freq[11] = 1850.000;   bandwidth[11]= 280.000;
    cent_freq[12] = 2150.000;   bandwidth[12]= 320.000;
    cent_freq[13] = 2500.000;   bandwidth[13]= 380.000;
    cent_freq[14] = 2900.000;   bandwidth[14]= 450.000;
    cent_freq[15] = 3400.000;   bandwidth[15]= 550.000;

    Weight=np.array([0.0192,0.0312,0.0926,0.1031,0.0735,0.0611,0.0495,0.044,0.044,0.049,0.0486,0.0493, 0.049,0.0547,0.0555,0.0493])
   
    # ----------------------------------------------------------------------
    # Set up the critical band filters.  Note here that Gaussianly shaped
    # filters are used.  Also, the sum of the filter weights are equivalent
    # for each critical band filter.  Filter less than -30 dB and set to
    # zero.
    # ----------------------------------------------------------------------

    all_f0=np.zeros((num_crit,))
    crit_filter=np.zeros((num_crit,int(n_fftby2)))
    g = np.zeros((num_crit,n_fftby2))
    
    b = bandwidth;
    q = cent_freq/1000;
    p = 4*1000*q/b;        # Eq. (7)
    
    #15.625=4000/256
    j = np.arange(0,n_fftby2)
    
    for i in range(num_crit):
        g[i,:]=np.abs(1-j*(sample_rate/n_fft)/(q[i]*1000));# Eq. (9)
        crit_filter[i,:] = (1+p[i]*g[i,:])*np.exp(-p[i]*g[i,:]);#  Eq. (8)

    num_frames = int(clean_length/skiprate-(winlength/skiprate)); # number of frames
    start      = 0 # starting sample
    hannWin = 0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    
    f,t,clean_spec=stft(clean_speech[0:int(num_frames)*skiprate+int(winlength-skiprate)], fs=fs, window=hannWin, nperseg=winlength, noverlap=winlength-skiprate, nfft=n_fft, detrend=False, return_onesided=False, boundary=None, padded=False)
    f,t,processed_spec=stft(processed_speech[0:int(num_frames)*skiprate+int(winlength-skiprate)], fs=fs, window=hannWin, nperseg=winlength, noverlap=winlength-skiprate, nfft=n_fft, detrend=False, return_onesided=False, boundary=None, padded=False)

    clean_frames = extractOverlappedWindows(clean_speech[0:int(num_frames)*skiprate+int(winlength-skiprate)],winlength,winlength-skiprate,None)
    rms_seg = np.linalg.norm(clean_frames,axis=-1)/np.sqrt(winlength);                                       
    rms_db = 20*np.log10(rms_seg/rms_all); 
    #--------------------------------------------------------------
    # cal r2_high,r2_middle,r2_low
    highInd = np.where(rms_db>=0)
    highInd = highInd[0]
    middleInd = np.where((rms_db>=-10) & (rms_db<0))
    middleInd = middleInd[0]
    lowInd = np.where(rms_db<-10)
    lowInd = lowInd[0]
    
    num_high = np.sum(clean_spec[0:n_fftby2,highInd]*np.conj(processed_spec[0:n_fftby2,highInd]),axis=-1)
    denx_high = np.sum(np.abs(clean_spec[0:n_fftby2,highInd])**2,axis=-1)
    deny_high = np.sum(np.abs(processed_spec[0:n_fftby2,highInd])**2,axis=-1);
    
    num_middle = np.sum(clean_spec[0:n_fftby2,middleInd]*np.conj(processed_spec[0:n_fftby2,middleInd]),axis=-1)
    denx_middle = np.sum(np.abs(clean_spec[0:n_fftby2,middleInd])**2,axis=-1)
    deny_middle = np.sum(np.abs(processed_spec[0:n_fftby2,middleInd])**2,axis=-1);

    num_low = np.sum(clean_spec[0:n_fftby2,lowInd]*np.conj(processed_spec[0:n_fftby2,lowInd]),axis=-1)
    denx_low = np.sum(np.abs(clean_spec[0:n_fftby2,lowInd])**2,axis=-1)
    deny_low = np.sum(np.abs(processed_spec[0:n_fftby2,lowInd])**2,axis=-1);
    
    num2_high = np.abs(num_high)**2;
    r2_high = num2_high/(denx_high*deny_high);

    num2_middle = np.abs(num_middle)**2;
    r2_middle = num2_middle/(denx_middle*deny_middle);

    num2_low = np.abs(num_low)**2;
    r2_low = num2_low/(denx_low*deny_low);
    #--------------------------------------------------------------
    # cal distortion frame by frame
        
    clean_spec     = np.abs(clean_spec);
    processed_spec = np.abs(processed_spec)**2; 

    W_freq=Weight
    
    processed_energy = crit_filter.dot((processed_spec[0:n_fftby2,highInd].T*r2_high).T)
    de_processed_energy= crit_filter.dot((processed_spec[0:n_fftby2,highInd].T*(1-r2_high)).T)
    SDR = processed_energy/de_processed_energy;# Eq 13 in Kates (2005)
    SDRlog=10*np.log10(SDR);
    SDRlog_lim = SDRlog   
    SDRlog_lim[SDRlog_lim<-15]=-15
    SDRlog_lim[SDRlog_lim>15]=15 # limit between [-15, 15]
    Tjm  = (SDRlog_lim+15)/30;    
    distortionh   =  W_freq.dot(Tjm)/np.sum(W_freq,axis=0)
    distortionh[distortionh<0]=0

    
    processed_energy = crit_filter.dot((processed_spec[0:n_fftby2,middleInd].T*r2_middle).T)
    de_processed_energy= crit_filter.dot((processed_spec[0:n_fftby2,middleInd].T*(1-r2_middle)).T)
    SDR = processed_energy/de_processed_energy;# Eq 13 in Kates (2005)
    SDRlog=10*np.log10(SDR);
    SDRlog_lim = SDRlog   
    SDRlog_lim[SDRlog_lim<-15]=-15
    SDRlog_lim[SDRlog_lim>15]=15 # limit between [-15, 15]
    Tjm  = (SDRlog_lim+15)/30;    
    distortionm   =  W_freq.dot(Tjm)/np.sum(W_freq,axis=0)
    distortionm[distortionm<0]=0
    
    processed_energy = crit_filter.dot((processed_spec[0:n_fftby2,lowInd].T*r2_low).T)
    de_processed_energy= crit_filter.dot((processed_spec[0:n_fftby2,lowInd].T*(1-r2_low)).T)
    SDR = processed_energy/de_processed_energy;# Eq 13 in Kates (2005)
    SDRlog=10*np.log10(SDR);
    SDRlog_lim = SDRlog   
    SDRlog_lim[SDRlog_lim<-15]=-15
    SDRlog_lim[SDRlog_lim>15]=15 # limit between [-15, 15]
    Tjm  = (SDRlog_lim+15)/30;    
    distortionl   =  W_freq.dot(Tjm)/np.sum(W_freq,axis=0)
    distortionl[distortionl<0]=0

    return distortionh,distortionm,distortionl


def CSII(clean_speech, processed_speech,sample_rate):
    sampleLen= min(len( clean_speech), len( processed_speech))
    clean_speech= clean_speech[0: sampleLen]
    processed_speech= processed_speech[0: sampleLen]
    vec_CSIIh,vec_CSIIm,vec_CSIIl = fwseg_noise(clean_speech, processed_speech, sample_rate)

    CSIIh=np.mean(vec_CSIIh)
    CSIIm=np.mean(vec_CSIIm)
    CSIIl=np.mean(vec_CSIIl)
    return CSIIh,CSIIm,CSIIl



def Get_Band(M,Fs):
    #   This function sets the bandpass filter band edges.
    # It assumes that the sampling frequency is 8000 Hz.
    A =   165
    a =   2.1
    K =   1
    L =   35
    CF = 300;
    x_100 =   (L/a)*np.log10(CF/A + K)
    CF = Fs/2-600
    x_8000 =   (L/a)*np.log10(CF/A + K);
    LX =   x_8000 - x_100
    x_step =   LX / M
    x = np.arange(x_100,x_8000+x_step+1e-20,x_step)
    if len(x) == M:
        np.append(x,x_8000)

    BAND = A*(10**(a*x/L) - K)
    return BAND

def get_ANSIs(BAND):
    fcenter=(BAND[0:-1]+BAND[1:])/2;

    # Data from Table B.1 in "ANSI (1997). S3.5–1997 Methods for Calculation of the Speech Intelligibility
    # Index. New York: American National Standards Institute."
    f=np.array([150,250,350,450,570,700,840,1000,1170,1370,1600,1850,2150,2500,2900,3400,4000,4800,5800,7000,8500])
    BIF=np.array([0.0192,0.0312,0.0926,0.1031,0.0735,0.0611,0.0495,0.0440,0.0440,0.0490,0.0486,0.0493,0.0490,0.0547,0.0555,0.0493,0.0359,0.0387,0.0256,0.0219,0.0043])
    f_ANSI = interp1d(f,BIF)
    ANSIs= f_ANSI(fcenter);
    return fcenter,ANSIs


def NCM(clean_speech,processed_speech,fs):

    x= clean_speech  # clean signal
    y= processed_speech # noisy signal
    F_SIGNAL = fs

    F_ENVELOPE  =   32 # limits modulations to 0<f<16 Hz      
    M_CHANNELS  =   20

    #   DEFINE BAND EDGES
    BAND = Get_Band(M_CHANNELS, F_SIGNAL);


    #   Interpolate the ANSI weights in WEIGHT @ fcenter
    fcenter,WEIGHT=get_ANSIs(BAND);

    #   NORMALIZE LENGTHS
    Lx          =   len(x);
    Ly          =   len(y);


    if Lx > Ly:
        x  = x[0:Ly]
    if Ly > Lx:
        y  = y[0:Lx]

    Lx          =   len(x);
    Ly          =   len(y);

    X_BANDS = np.zeros((Lx,M_CHANNELS))
    Y_BANDS = np.zeros((Lx,M_CHANNELS))

    #   DESIGN BANDPASS FILTERS
    for a in range(M_CHANNELS):
        B_bp,A_bp  = butter( 4 , np.array([BAND[a],BAND[a+1]])*(2/F_SIGNAL),btype='bandpass')
        X_BANDS[:,a] = lfilter( B_bp , A_bp , x )
        Y_BANDS[:,a] = lfilter( B_bp , A_bp , y )

    #   CALCULATE HILBERT ENVELOPES, and resample at F_ENVELOPE Hz
    analytic_x = hilbert( X_BANDS,axis=0);
    X   = np.abs( analytic_x );
    X   = resample( X , round(len(x)/F_SIGNAL*F_ENVELOPE));

    analytic_y = hilbert( Y_BANDS,axis=0);
    Y = np.abs( analytic_y );
    Y = resample( Y , round(len(x)/F_SIGNAL*F_ENVELOPE));

    ## ---compute weights based on clean signal's rms envelopes-----
    #
    Ldx, pp=X.shape
    p=3 # power exponent - see Eq. 12

    ro2 = np.zeros((M_CHANNELS,))
    asnr = np.zeros((M_CHANNELS,))
    TI = np.zeros((M_CHANNELS,))

    for k in range(M_CHANNELS):
        x_tmp= X[ :, k]
        y_tmp= Y[ :, k]
        lambda_x= np.linalg.norm( x_tmp- np.mean( x_tmp))**2
        lambda_y= np.linalg.norm( y_tmp- np.mean( y_tmp))**2
        lambda_xy= np.sum( (x_tmp- np.mean( x_tmp))*(y_tmp- np.mean( y_tmp)))
        ro2[k]= (lambda_xy**2)/ (lambda_x*lambda_y)
        asnr[k]= 10*np.log10( (ro2[k]+ 1e-20)/ (1- ro2[k]+ 1e-20)); # Eq.9 in [1]

        if asnr[k]< -15:
            asnr[k]= -15
        elif asnr[k]> 15:
            asnr[k]= 15

        TI[k]= (asnr[k]+ 15)/ 30 # Eq.10 in [1]

    ncm_val= WEIGHT.dot(TI)/np.sum(WEIGHT) # Eq.11
    return ncm_val