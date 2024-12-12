from scipy import signal as sgn
from scipy.fft import fftshift
import numpy as np

def Spectogram(data:np.ndarray):
    f, t, Sxx = sgn.spectrogram(data, fs=128.0, window=('boxcar'), nperseg=600.0, noverlap=500.0, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')
    return f,t,Sxx