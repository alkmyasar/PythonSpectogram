from readCsv import readSensorConnectCsv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square, ShortTimeFFT
from scipy.signal.windows import gaussian, boxcar


filename = "data/Kopruagzi/EK-R-875/ch1.csv"

data  = readSensorConnectCsv(filename)
data = np.array(data[:,1])
data = data - np.mean(data)

windowSize = 128*3
hopsize = 10
mfftnum = 128*8

# t = np.linspace(0,100,128*100+1)
# data = np.sin(2*np.pi*20*t)+8*np.sin(2*np.pi*8*t)+12*np.sin(2*np.pi*4.9658*t)

T_x, N = 1 / 128, data.shape[0]

g_std = 12  # standard deviation for Gaussian window in samples

# win = gaussian(500, std=g_std, sym=True)  # symmetric Gaussian wind.
win = boxcar(windowSize, sym=True)

SFT = ShortTimeFFT(win, hop=hopsize, fs=128, mfft=mfftnum, scale_to='psd')

Sx2 = SFT.spectrogram(data)  # calculate absolute square of STFT


fig1, ax1 = plt.subplots(figsize=(6., 4.))  # enlarge plot a bit

t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot

ax1.set_title(f"Spectrogram ws = {windowSize} hop = {hopsize} mfft = {mfftnum}")

ax1.set(xlabel=f"Time $t$ in seconds" +

               rf"$\Delta t = {SFT.delta_t:g}\,$s)",

        ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +

               rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",

        xlim=(t_lo, t_hi))

#Sx_dB = 10 * np.log10(np.fmax(Sx2, 1e-11))  # limit range to -40 dB
Sx_dB = 10 * np.log10(np.clip(Sx2,1e-11,1e-6))
#Sx_dB = 10 * np.log10(np.clip(Sx2,1e-11,1e-3))
im1 = ax1.imshow(Sx_dB, origin='lower', aspect='auto',

                 extent=SFT.extent(N), cmap='magma')


fig1.colorbar(im1, label='Power Spectral Density ' +

                         r"$20\,\log_{10}|S_x(t, f)|$ in dB")


# Shade areas where window slices stick out to the side:

for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),

                 (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:

    ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.3)

for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line

    ax1.axvline(t_, color='c', linestyle='--', alpha=0.5)


fig1.tight_layout()


plt.show()
