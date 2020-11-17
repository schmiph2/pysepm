from scipy.signal import stft,get_window,correlate,resample
from scipy.linalg import solve_toeplitz,toeplitz
import scipy
import pesq as pypesq # https://github.com/ludlows/python-pesq
import numpy as np
from numba import jit
from .util import extract_overlapped_windows

def SNRseg(clean_speech, processed_speech,fs, frameLen=0.03, overlap=0.75):
    eps=np.finfo(np.float64).eps

    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples
    MIN_SNR     = -10 # minimum SNR in dB
    MAX_SNR     =  35 # maximum SNR in dB

    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    clean_speech_framed=extract_overlapped_windows(clean_speech,winlength,winlength-skiprate,hannWin)
    processed_speech_framed=extract_overlapped_windows(processed_speech,winlength,winlength-skiprate,hannWin)
    
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
    cent_freq[9] =  703.378;   bandwidth[9] =  95.3398;
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
@jit
def lpcoeff(speech_frame, model_order):
    eps=np.finfo(np.float64).eps
   # ----------------------------------------------------------
   # (1) Compute Autocorrelation Lags
   # ----------------------------------------------------------
    winlength = max(speech_frame.shape)
    R = np.zeros((model_order+1,))
    for k in range(model_order+1):
        if k==0:
            R[k]=np.sum(speech_frame[0:]*speech_frame[0:])
        else:
            R[k]=np.sum(speech_frame[0:-k]*speech_frame[k:])
        
     
    #R=scipy.signal.correlate(speech_frame,speech_frame) 
    #R=R[len(speech_frame)-1:len(speech_frame)+model_order]
   # ----------------------------------------------------------
   # (2) Levinson-Durbin
   # ----------------------------------------------------------
    a = np.ones((model_order,))
    a_past = np.ones((model_order,))
    rcoeff = np.zeros((model_order,))
    E = np.zeros((model_order+1,))

    E[0]=R[0]

    for i in range(0,model_order):
        a_past[0:i] = a[0:i]

        sum_term = np.sum(a_past[0:i]*R[i:0:-1])
		
        if E[i]==0.0: # prevents zero division error, numba doesn't allow try/except statements
            rcoeff[i]= np.inf
        else:
            rcoeff[i]=(R[i+1] - sum_term) / (E[i])
			
        a[i]=rcoeff[i]
        #if i==0:
        #    a[0:i] = a_past[0:i] - rcoeff[i]*np.array([])
        #else:
        if i>0:
            a[0:i] = a_past[0:i] - rcoeff[i]*a_past[i-1::-1]

        E[i+1]=(1-rcoeff[i]*rcoeff[i])*E[i]

    acorr    = R;
    refcoeff = rcoeff;
    lpparams = np.ones((model_order+1,))
    lpparams[1:] = -a
    return(lpparams,R)

def llr(clean_speech, processed_speech, fs, used_for_composite=False, frameLen=0.03, overlap=0.75):
    eps=np.finfo(np.float64).eps
    alpha = 0.95
    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples    
    if fs<10000:
        P = 10 # LPC Analysis Order
    else:
        P = 16 # this could vary depending on sampling frequency.
        
    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    clean_speech_framed=extract_overlapped_windows(clean_speech+eps,winlength,winlength-skiprate,hannWin)
    processed_speech_framed=extract_overlapped_windows(processed_speech+eps,winlength,winlength-skiprate,hannWin)
    numFrames=clean_speech_framed.shape[0]
    numerators = np.zeros((numFrames-1,))
    denominators = np.zeros((numFrames-1,))
    
    for ii in range(numFrames-1):
        A_clean,R_clean=lpcoeff(clean_speech_framed[ii,:],P)
        A_proc,R_proc=lpcoeff(processed_speech_framed[ii,:],P)
        
        numerators[ii]=A_proc.dot(toeplitz(R_clean).dot(A_proc.T))
        denominators[ii]=A_clean.dot(toeplitz(R_clean).dot(A_clean.T))
    
    
    frac=numerators/(denominators)
    frac[np.isnan(frac)]=np.inf
    frac[frac<=0]=1000
    distortion = np.log(frac)
    if not used_for_composite:
        distortion[distortion>2]=2 # this line is not in composite measure but in llr matlab implementation of loizou
    distortion = np.sort(distortion)
    distortion = distortion[:int(round(len(distortion)*alpha))]
    return np.mean(distortion)


@jit
def find_loc_peaks(slope,energy):
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
        clean_loc_peaks[:,ii]=find_loc_peaks(log_clean_energy_slope[:,ii],log_clean_energy[:,ii])
        proc_loc_peaks[:,ii]=find_loc_peaks(log_proc_energy_slope[:,ii],log_proc_energy[:,ii])
    

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
        mos_lqo = pypesq.pesq(fs,clean_speech, processed_speech, 'nb')
        pesq_mos = 46607/14945 - (2000*np.log(1/(mos_lqo/4 - 999/4000) - 1))/2989#0.999 + ( 4.999-0.999 ) / ( 1+np.exp(-1.4945*pesq_mos+4.6607) )
    elif fs == 16000:
        mos_lqo = pypesq.pesq(fs,clean_speech, processed_speech, 'wb')
        pesq_mos = np.NaN
    else:
        raise ValueError('fs must be either 8 kHz or 16 kHz')
        
    return pesq_mos,mos_lqo


def composite(clean_speech, processed_speech, fs):
    wss_dist=wss(clean_speech, processed_speech, fs)
    llr_mean=llr(clean_speech, processed_speech, fs,used_for_composite=True)
    segSNR=SNRseg(clean_speech, processed_speech, fs)
    pesq_mos,mos_lqo = pesq(clean_speech, processed_speech,fs)
    
    if fs >= 16e3:
        used_pesq_val = mos_lqo
    else:
        used_pesq_val = pesq_mos    

    Csig = 3.093 - 1.029*llr_mean + 0.603*used_pesq_val-0.009*wss_dist
    Csig = np.max((1,Csig))  
    Csig = np.min((5, Csig)) # limit values to [1, 5]
    Cbak = 1.634 + 0.478 *used_pesq_val - 0.007*wss_dist + 0.063*segSNR
    Cbak = np.max((1, Cbak))
    Cbak = np.min((5,Cbak)) # limit values to [1, 5]
    Covl = 1.594 + 0.805*used_pesq_val - 0.512*llr_mean - 0.007*wss_dist
    Covl = np.max((1, Covl))
    Covl = np.min((5, Covl)) # limit values to [1, 5]
    return Csig,Cbak,Covl

@jit
def lpc2cep(a):
    #
    # converts prediction to cepstrum coefficients
    #
    # Author: Philipos C. Loizou

    M=len(a);
    cep=np.zeros((M-1,));

    cep[0]=-a[1]

    for k in range(2,M):
        ix=np.arange(1,k)
        vec1=cep[ix-1]*a[k-1:0:-1]*(ix)
        cep[k-1]=-(a[k]+np.sum(vec1)/k);
    return cep


def cepstrum_distance(clean_speech, processed_speech, fs, frameLen=0.03, overlap=0.75):
    
    
    clean_length      = len(clean_speech)
    processed_length  = len(processed_speech)

    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples
    
    if fs<10000:
        P = 10 # LPC Analysis Order
    else:
        P=16;    # this could vary depending on sampling frequency.

    C=10*np.sqrt(2)/np.log(10)

    numFrames = int(clean_length/skiprate-(winlength/skiprate)); # number of frames

    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    clean_speech_framed=extract_overlapped_windows(clean_speech[0:int(numFrames)*skiprate+int(winlength-skiprate)],winlength,winlength-skiprate,hannWin)
    processed_speech_framed=extract_overlapped_windows(processed_speech[0:int(numFrames)*skiprate+int(winlength-skiprate)],winlength,winlength-skiprate,hannWin)   
    distortion = np.zeros((numFrames,))

    for ii in range(numFrames):
        A_clean,R_clean=lpcoeff(clean_speech_framed[ii,:],P)
        A_proc,R_proc=lpcoeff(processed_speech_framed[ii,:],P)

        C_clean=lpc2cep(A_clean)
        C_processed=lpc2cep(A_proc)
        distortion[ii] = min((10,C*np.linalg.norm(C_clean-C_processed)))
    
    IS_dist = distortion
    alpha=0.95
    IS_len= round( len( IS_dist)* alpha)
    IS = np.sort(IS_dist)
    cep_mean= np.mean( IS[ 0: IS_len]) 
    return cep_mean