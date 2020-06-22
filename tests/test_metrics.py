from scipy.io import wavfile,loadmat
import numpy as np
import sys
sys.path.append("../") 
import pysepm as pm
import pytest
import numpy.testing
import librosa

RTOL = 1e-5
ATOL = 1e-4

resultsOrig = loadmat("results.mat")
resultsOrig=resultsOrig['results']

testScenarios=[]
pairCounter = 0
testScenarios.append((['speech.wav','speech_bab_0dB.wav'],resultsOrig[pairCounter,:]))
pairCounter = pairCounter + 1
testScenarios.append((['sp04.wav','sp04_babble_sn10.wav'],resultsOrig[pairCounter,:]))
pairCounter = pairCounter + 1
testScenarios.append((['enhanced.wav','sp04_babble_sn10.wav'],resultsOrig[pairCounter,:]))
pairCounter = pairCounter + 1
testScenarios.append((['sp04.wav','enhanced.wav'],resultsOrig[pairCounter,:]))
pairCounter = pairCounter + 1
testScenarios.append((['S_03_01.wav','S_03_01_babble_sn0_klt.wav'],resultsOrig[pairCounter,:]))
pairCounter = pairCounter + 1
testScenarios.append((['cleanSample_valIdx12.wav','noisySample_valIdx12.wav'],resultsOrig[pairCounter,:]))
pairCounter = pairCounter + 1
testScenarios.append((['cleanSample_valIdx12.wav','enhancedSample_valIdx12.wav'],resultsOrig[pairCounter,:]))
pairCounter = pairCounter + 1
testScenarios.append((['cleanSample_valIdx356.wav','noisySample_valIdx356.wav'],resultsOrig[pairCounter,:]))
pairCounter = pairCounter + 1
testScenarios.append((['cleanSample_valIdx356.wav','enhancedSample_valIdx356.wav'],resultsOrig[pairCounter,:]))


def load_preprocess_filepair(filePair,resample=False,fs_targ=16000):
    fs, cleanSig = wavfile.read('data/'+filePair[0])
    fs, enhancedSig = wavfile.read('data/'+filePair[1])
    if cleanSig.dtype=='int16':
        cleanSig = cleanSig.astype('float')/abs(np.iinfo(cleanSig.dtype).min)
    if enhancedSig.dtype=='int16':
        enhancedSig = enhancedSig.astype('float')/abs(np.iinfo(enhancedSig.dtype).min)
    
    if cleanSig.shape!=enhancedSig.shape:
        minLen=min((len(cleanSig),len(enhancedSig)))
        cleanSig = cleanSig[0:minLen]
        enhancedSig = enhancedSig[0:minLen]
    cleanSig = cleanSig/np.max(np.abs(cleanSig))
    enhancedSig = enhancedSig/np.max(np.abs(enhancedSig))
    
    if resample:
        fs, cleanSig = wavfile.read('data/'+str(int(fs_targ/1000))+'kHz_'+filePair[0])
        fs, enhancedSig = wavfile.read('data/'+str(int(fs_targ/1000))+'kHz_'+filePair[1])

        #cleanSig=librosa.core.resample(cleanSig, fs,fs_targ)
        #enhancedSig=librosa.core.resample(enhancedSig,fs,fs_targ)        
        #fs = fs_targ        

    return cleanSig,enhancedSig,fs

@pytest.mark.parametrize('filePair,expected_vals', testScenarios)
def test_fwSNRseg(filePair,expected_vals):
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair)
    numpy.testing.assert_allclose(pm.fwSNRseg(cleanSig, enhancedSig, fs), expected_vals[0], rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize('filePair,expected_vals', testScenarios)
def test_SNRseg(filePair,expected_vals):
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair)
    numpy.testing.assert_allclose(pm.SNRseg(cleanSig, enhancedSig, fs), expected_vals[1], rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize('filePair,expected_vals', testScenarios)
def test_llr(filePair,expected_vals):
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair)
    numpy.testing.assert_allclose(pm.llr(cleanSig, enhancedSig, fs), expected_vals[2], rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize('filePair,expected_vals', testScenarios)
def test_wss(filePair,expected_vals):
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair)
    numpy.testing.assert_allclose(pm.wss(cleanSig, enhancedSig, fs), expected_vals[3], rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize('filePair,expected_vals', testScenarios)
def test_cepstrum_distance(filePair,expected_vals):
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair)
    numpy.testing.assert_allclose(pm.cepstrum_distance(cleanSig, enhancedSig, fs), expected_vals[4], rtol=RTOL, atol=ATOL)
    
@pytest.mark.parametrize('filePair,expected_vals', testScenarios) # needs 10 kHz samping rate!
def test_stoi(filePair,expected_vals):
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair,True,10e3)
    
    numpy.testing.assert_allclose(pm.stoi(cleanSig, enhancedSig, fs), expected_vals[5], rtol=RTOL, atol=ATOL)
    
@pytest.mark.parametrize('filePair,expected_vals', testScenarios) # needs 10 kHz samping rate!
def test_csii(filePair,expected_vals):
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair)
    CSIIh,CSIIm,CSIIl=pm.csii(cleanSig, enhancedSig, fs)
    numpy.testing.assert_allclose(CSIIh, expected_vals[6], rtol=RTOL, atol=ATOL)
    numpy.testing.assert_allclose(CSIIm, expected_vals[7], rtol=RTOL, atol=ATOL)
    numpy.testing.assert_allclose(CSIIl, expected_vals[8], rtol=RTOL, atol=ATOL)

    
@pytest.mark.parametrize('filePair,expected_vals', testScenarios) # needs 10 kHz samping rate!
def test_pesq(filePair,expected_vals):
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair,True,16e3)
    mos_lqo, pesq_mos=pm.pesq(cleanSig, enhancedSig, fs)
    numpy.testing.assert_allclose(mos_lqo, expected_vals[10], rtol=RTOL, atol=ATOL)

@pytest.mark.parametrize('filePair,expected_vals', testScenarios) # needs 10 kHz samping rate!
def test_composite(filePair,expected_vals):
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair,True,16e3)
    Csig,Cbak,Covl=pm.composite(cleanSig, enhancedSig, fs)
    numpy.testing.assert_allclose(Csig, expected_vals[11], rtol=RTOL, atol=ATOL)
    numpy.testing.assert_allclose(Cbak, expected_vals[12], rtol=RTOL, atol=ATOL)
    numpy.testing.assert_allclose(Covl, expected_vals[13], rtol=RTOL, atol=ATOL)
    
@pytest.mark.parametrize('filePair,expected_vals', testScenarios) # needs 10 kHz samping rate!
def test_ncm(filePair,expected_vals):
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair,True,16e3)
    numpy.testing.assert_allclose(pm.ncm(cleanSig, enhancedSig, fs), expected_vals[14], rtol=RTOL, atol=ATOL)

    
@pytest.mark.parametrize('filePair,expected_vals', testScenarios) # needs 10 kHz samping rate!
def test_srmr(filePair,expected_vals):
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair,True,16e3)
    ratio,energy = pm.srmr(enhancedSig, fs)
    numpy.testing.assert_allclose(ratio, expected_vals[15], rtol=RTOL, atol=ATOL)

    
@pytest.mark.parametrize('filePair,expected_vals', testScenarios) # needs 10 kHz samping rate!
def test_bsd(filePair,expected_vals):
    cleanSig,enhancedSig,fs=load_preprocess_filepair(filePair)
    ratio,energy = pm.bsd(cleanSig, enhancedSig, fs)
    numpy.testing.assert_allclose(ratio, np.inf, rtol=RTOL, atol=ATOL)
