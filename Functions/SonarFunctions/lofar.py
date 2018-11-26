from __future__ import division
from scipy.signal import decimate, hanning, convolve, spectrogram
import numpy as np
#from simplespectral import spectrogram


def tpsw(x, npts=None, n=None, p=None, a=None):
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if npts is None:
        npts = x.shape[0]
    if n is None:
        n=int(round(npts*.04/2.0+1))
    if p is None:
        p =int(round(n / 8.0 + 1))
    if a is None:
        a = 2.0
    if p>0:
        h = np.concatenate((np.ones((n-p+1)), np.zeros(2 * p-1), np.ones((n-p+1))), axis=None)
    else:
        h = np.ones((1, 2*n+1))
        p = 1
    h /= np.linalg.norm(h, 1)

    def apply_on_spectre(xs):
        return convolve(h, xs, mode='full')

    mx = np.apply_along_axis(apply_on_spectre, arr=x, axis=0)
    ix = int(np.floor((h.shape[0] + 1)/2.0)) # Defasagem do filtro
    mx = mx[ix-1:npts+ix-1] # Corrige da defasagem
    # Corrige os pontos extremos do espectro
    ixp = ix - p
    mult=2*ixp/np.concatenate([np.ones(p-1)*ixp, range(ixp,2*ixp + 1)], axis=0)[:, np.newaxis] # Correcao dos pontos extremos
    mx[:ix,:] = mx[:ix,:]*(np.matmul(mult, np.ones((1, x.shape[1])))) # Pontos iniciais
    mx[npts-ix:npts,:]=mx[npts-ix:npts,:]*np.matmul(np.flipud(mult),np.ones((1, x.shape[1]))) # Pontos finais
    #return mx
    # Elimina picos para a segunda etapa da filtragem
    #indl= np.where((x-a*mx) > 0) # Pontos maiores que a*mx
    indl = (x-a*mx) > 0
    #x[indl] = mx[indl]
    x = np.where(indl, mx, x)
    mx = np.apply_along_axis(apply_on_spectre, arr=x, axis=0)
    mx=mx[ix-1:npts+ix-1,:]
    #Corrige pontos extremos do espectro
    mx[:ix,:]=mx[:ix,:]*(np.matmul(mult,np.ones((1, x.shape[1])))) # Pontos iniciais
    mx[npts-ix:npts,:]=mx[npts-ix:npts,:]*(np.matmul(np.flipud(mult),np.ones((1,x.shape[1])))) # Pontos finais
    return mx

def lofar(data, fs, n_pts_fft=1024, n_overlap=0,
    decimation_rate=3, spectrum_bins_left=None, **tpsw_args):
    norm_parameters = {'lat_window_size': 10, 'lat_gap_size': 1, 'threshold': 1.3}
    if not isinstance(data, np.ndarray):
        raise NotImplementedError

    if decimation_rate > 1:
        #dec_data = decimate(data, decimation_rate, 10, 'fir', zero_phase=True)
        Fs = fs/decimation_rate
    else:
        dec_data = data
        Fs=fs
    freq, time, power = spectrogram(data,
                                    window=('hann'),
                                    nperseg=n_pts_fft,
                                    noverlap=n_overlap,
                                    nfft=n_pts_fft,
                                    fs=Fs,
                                    axis=0,
                                    scaling='density',
                                    mode='complex')
    #power = np.absolute(power)
    #power = power / tpsw(power)#, **tpsw_args)
    #power = np.log10(power)
    #power[power < -0.2] = 0

    if spectrum_bins_left is None:
        spectrum_bins_left = power.shape[0]*0.8
    power = power[:spectrum_bins_left, :]
    freq = freq[:spectrum_bins_left]

    return power, freq, time


