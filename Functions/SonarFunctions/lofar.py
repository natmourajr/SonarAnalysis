from cmath import log10
from scipy.signal import decimate, hanning, spectrogram, convolve2d
import numpy as np


def tpsw(x, npts=None, n=None, p=None, a=None):
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if npts is None:
        npts = x.shape[0]
    if n is None:
        n=round(npts*.04/2+1)
    if p is None:
        p = round(n / 8 + 1)
    if a is None:
        a = 2.0

    if p>0:
        h = np.concatenate([np.ones(1, n - p + 1), np.zeros(1, 2 * p - 1), np.ones(1, n - p + 1)])
    else:
        h = [np.ones(1, 2 * n + 1)]
        p = 1

    h = h[:, np.newaxis]

    h = h/abs(h)
    mx = convolve2d(h, x)
    ix = np.floor((h.shape[0] + 1)/2.0) - 1
    mx = mx[:, ix:npts+ix-1]

    # Corrige os pontos extremos do espectro
    raise NotImplementedError


def lofar(data, fs, n_pts_fft=1024, n_overlap=0, decimation_rate=3, spectrum_bins_left=400):
    norm_parameters = {'lat_window_size': 10, 'lat_gap_size': 1, 'threshold': 1.3}

    if decimation_rate > 1:
        dec_data = decimate(data, decimation_rate, 10, 'fir')
        Fs = fs/decimation_rate
    else:
        dec_data = data
        Fs=fs

    freq, time, power = spectrogram(data,
                                    window=hanning(n_pts_fft),
                                    noverlap=n_overlap,
                                    nfft=n_pts_fft,
                                    fs=Fs)
    power=abs(power)
    power = power / tpsw(power)
    power = log10(power)
    power[power < -0.2] = 0

    power = power[:spectrum_bins_left, :]
    freq = freq[:spectrum_bins_left, :]


