from scipy.io import wavfile,loadmat
import numpy as np
import sys
sys.path.append("../") 
import pysepm as pm
import pytest
import numpy.testing
import librosa

RTOL = 1e-12
ATOL = 0

resultsOrig = loadmat("results.mat")
resultsOrig=resultsOrig['results']



freqs = [44100,22050,16000,8000]
pairs =  [['speech','noisySpeech'],['speech','processed']]
testScenarios=[]
pairCounter = 0
for ii in range(1,4):
    for f in freqs:
        for pair in pairs:
            fileNameClean="{}_{}_{}_Hz.wav".format(ii,pair[0],f)
            fileNameNoisy="{}_{}_{}_Hz.wav".format(ii,pair[1],f)
            testScenarios.append(([fileNameClean,fileNameNoisy],resultsOrig[pairCounter]))
            pairCounter=pairCounter+1


def load_preprocess_filepair(filePair,resample=False,fs_targ=16000):
    fs, cleanSig = wavfile.read('data/'+filePair[0])
    fs, enhancedSig = wavfile.read('data/'+filePair[1])
        
    if resample:
        if fs_targ == 16000 and fs !=8000:
            strParts=filePair[0].split('_')
            fileNameClean="{}_{}_{}_Hz.wav".format(strParts[0],strParts[1],16000)
            fs, cleanSig = wavfile.read('data/'+fileNameClean)

            strParts=filePair[1].split('_')
            fileNameNoisy="{}_{}_{}_Hz.wav".format(strParts[0],strParts[1],16000)
            fs, enhancedSig = wavfile.read('data/'+fileNameNoisy)

        elif fs_targ == 16000 and fs == 8000:
            pass
        else:
            
            cleanSig = librosa.core.resample(cleanSig, fs, fs_targ)
            enhancedSig = librosa.core.resample(enhancedSig, fs, fs_targ)
            fs = fs_targ

    return cleanSig,enhancedSig,fs

@pytest.mark.parametrize('filePair,expected_vals', testScenarios)
def test_fwSNRseg(filePair,expected_vals):
    RTOL = 1e-12
    ATOL = 0

    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair)
    numpy.testing.assert_allclose(pm.fwSNRseg(cleanSig, enhancedSig, fs), expected_vals[0], rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize('filePair,expected_vals', testScenarios)
def test_SNRseg(filePair,expected_vals):
    RTOL = 1e-12
    ATOL = 0
    
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair)
    numpy.testing.assert_allclose(pm.SNRseg(cleanSig, enhancedSig, fs), expected_vals[1], rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize('filePair,expected_vals', testScenarios)
def test_llr(filePair,expected_vals):
    RTOL = 5e-8
    ATOL = 0

    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair)
    numpy.testing.assert_allclose(pm.llr(cleanSig, enhancedSig, fs), expected_vals[2], rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize('filePair,expected_vals', testScenarios)
def test_wss(filePair,expected_vals):
    RTOL = 1e-12
    ATOL = 0

    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair)
    numpy.testing.assert_allclose(pm.wss(cleanSig, enhancedSig, fs), expected_vals[3], rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize('filePair,expected_vals', testScenarios)
def test_cepstrum_distance(filePair,expected_vals):
    RTOL = 1e-8
    ATOL = 0
    
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair)
    numpy.testing.assert_allclose(pm.cepstrum_distance(cleanSig, enhancedSig, fs), expected_vals[4], rtol=RTOL, atol=ATOL)
    
@pytest.mark.parametrize('filePair,expected_vals', testScenarios)
def test_stoi(filePair,expected_vals):
    RTOL = 5e-3#5e-4
    ATOL = 0

    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair,True,10e3)
    
    numpy.testing.assert_allclose(pm.stoi(cleanSig, enhancedSig, fs), expected_vals[5], rtol=RTOL, atol=ATOL)
    
@pytest.mark.parametrize('filePair,expected_vals', testScenarios) 
def test_csii(filePair,expected_vals):
    RTOL = 5e-4
    ATOL = 0

    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair)
    CSIIh,CSIIm,CSIIl=pm.csii(cleanSig, enhancedSig, fs)
    numpy.testing.assert_allclose(CSIIh, expected_vals[6], rtol=RTOL, atol=ATOL)
    numpy.testing.assert_allclose(CSIIm, expected_vals[7], rtol=RTOL, atol=ATOL)
    numpy.testing.assert_allclose(CSIIl, expected_vals[8], rtol=RTOL, atol=ATOL)

    
@pytest.mark.parametrize('filePair,expected_vals', testScenarios) 
def test_pesq(filePair,expected_vals):
    RTOL = 5e-4
    ATOL = 0
    
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair,True,16e3)
    pesq_mos,mos_lqo=pm.pesq(cleanSig, enhancedSig, fs)
    numpy.testing.assert_allclose(mos_lqo, expected_vals[10], rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize('filePair,expected_vals', testScenarios) 
def test_composite(filePair,expected_vals):
    RTOL = 5e-4
    ATOL = 0

    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair,True,16e3)
    Csig,Cbak,Covl=pm.composite(cleanSig, enhancedSig, fs)
    numpy.testing.assert_allclose(Csig, expected_vals[11], rtol=RTOL, atol=ATOL)
    numpy.testing.assert_allclose(Cbak, expected_vals[12], rtol=RTOL, atol=ATOL)
    numpy.testing.assert_allclose(Covl, expected_vals[13], rtol=RTOL, atol=ATOL)
    
@pytest.mark.parametrize('filePair,expected_vals', testScenarios)
def test_ncm(filePair,expected_vals):
    RTOL = 5e-6
    ATOL = 0
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair,True,16e3)
    numpy.testing.assert_allclose(pm.ncm(cleanSig, enhancedSig, fs), expected_vals[14], rtol=RTOL, atol=ATOL)

    
@pytest.mark.parametrize('filePair,expected_vals', testScenarios)
def test_srmr(filePair,expected_vals):
    RTOL = 5e-4
    ATOL = 0
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair,True,16e3)
    ratio = pm.srmr(enhancedSig, fs)
    numpy.testing.assert_allclose(ratio, expected_vals[15], rtol=RTOL, atol=ATOL)

    
#@pytest.mark.parametrize('filePair,expected_vals', testScenarios)
#def test_bsd(filePair,expected_vals):
#    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair)
#    bsd = pm.bsd(cleanSig, enhancedSig, fs)
#    numpy.testing.assert_allclose(bsd, 1e20, rtol=RTOL, atol=ATOL)
