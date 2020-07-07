import numpy as np
from scipy.signal import firls,kaiser,upfirdn
from fractions import Fraction

def extract_overlapped_windows(x,nperseg,noverlap,window=None):
    # source: https://github.com/scipy/scipy/blob/v1.2.1/scipy/signal/spectral.py
    step = nperseg - noverlap
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    if window is not None:
        result = window * result
    return result

def resample_matlab_like(x_orig,p,q):
    if len(x_orig.shape)>2:
        raise ValueError('x must be a vector or 2d matrix')
        
    if x_orig.shape[0]<x_orig.shape[1]:
        x=x_orig.T
    else:
        x= x_orig
    beta = 5
    N = 10
    frac=Fraction(p, q)
    p = frac.numerator
    q = frac.denominator
    pqmax = max(p,q)
    fc = 1/2/pqmax
    L = 2*N*pqmax + 1
    h = firls( L, np.array([0,2*fc,2*fc,1]), np.array([1,1,0,0]))*kaiser(L,beta)
    h = p*h/sum(h)

    Lhalf = (L-1)/2
    Lx = x.shape[0]

    nz = int(np.floor(q-np.mod(Lhalf,q)))
    z = np.zeros((nz,))
    h = np.concatenate((z,h))
    Lhalf = Lhalf + nz
    delay = int(np.floor(np.ceil(Lhalf)/q))
    nz1 = 0
    while np.ceil( ((Lx-1)*p+len(h)+nz1 )/q ) - delay < np.ceil(Lx*p/q):
        nz1 = nz1+1
    h = np.concatenate((h,np.zeros(nz1,)))
    y = upfirdn(h,x,p,q,axis=0)
    Ly = int(np.ceil(Lx*p/q))
    y = y[delay:]
    y = y[:Ly]
    
    if x_orig.shape[0]<x_orig.shape[1]:
        y=y.T
    
    return y