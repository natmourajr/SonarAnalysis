from __future__ import division, print_function

import os
import numpy as np
import pandas as pd
import matplotlib
import sys
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from scipy.io import loadmat
from sklearn.neighbors import KernelDensity
from sklearn.externals import joblib
from Functions.DataHandler import LofarDataset
from Functions.SonarFunctions.read_raw_data import AudioData
from Functions.FunctionsDataVisualization import plotSpectrogram
from Functions.SonarFunctions.lofar_analysis import LofarAnalysis, tpsw

from multiprocessing import Pool

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rc('legend', **{'fontsize': 15})
plt.rc('font', weight='bold')


datapath = os.getenv('OUTPUTDATAPATH')
audiodatapath = os.getenv('INPUTDATAPATH')
results_path = os.getenv('PACKAGE_NAME')
database = '4classes'

lofar = LofarDataset(datapath)
window=int(sys.argv[1]); overlap=int(sys.argv[2]); decimation_rate=int(sys.argv[3]); spectrum_bins_left=int(sys.argv[4])

ad = AudioData(audiodatapath, database)
raw_data,fs = ad.read_raw_data(verbose=0)

raw_data_seg = raw_data.copy()
for cls in raw_data:
    for run_name, run in raw_data[cls].items():
        seg_run = np.zeros((run.shape[0]//(window-overlap), window))
        
        for i in range(seg_run.shape[0]):
            seg_run[i,:] = run[i*(window):(i*(window - overlap) + window)]
    
        raw_data_seg[cls][run_name] = seg_run

kernel = 'gaussian'
bins = np.linspace(-1,1,200)
kde_path = './kde_%i_%i_%i_%i.jbl' % (window, overlap, 
                                             decimation_rate, spectrum_bins_left)
if os.path.exists(kde_path):
    kdes = joblib.load(kde_path)
else:
    kdes = dict()
    for cls in raw_data_seg:
        kdes[cls] = dict()
        for run_name, run_seg in raw_data_seg[cls].items():
            kdes[cls][run_name] = np.zeros((run_seg.shape[0], bins.shape[0]))
            for i in range(0, run_seg.shape[0]):
                segment = run_seg[i]
                kde = KernelDensity(kernel=kernel, 
                                     bandwidth=0.5).fit(segment[:, np.newaxis])
                kdes[cls][run_name][i] = np.exp(kde.score_samples(bins[:, np.newaxis]))
    print(time.time() - start)
    joblib.dump(kdes, kde_path)
    
from Functions.StatFunctions import KLDiv
def kl_dv_fn(kdes, k):
    kl_foward = {}
    kl_reverse = {}
    for cls in kdes:
        kl_foward[cls] = dict()
        kl_reverse[cls] = dict()
        for run in kdes[cls]:
            run_pdf = kdes[cls][run]
            kl_foward[cls][run] = np.zeros(run_pdf.shape[0] - 1)
            kl_reverse[cls][run] = np.zeros(run_pdf.shape[0] - 1)
            for i in range(k,kl_foward[cls][run].shape[0]):
                kl_foward[cls][run][i-k] = np.absolute(KLDiv(run_pdf[i-k], run_pdf[i])[0] )
                kl_reverse[cls][run][i-k] = np.absolute(KLDiv(run_pdf[i], run_pdf[i-k])[0])
    
    kldiv_foward = dict()
    kldiv_reverse = dict()
    for cls in kdes:
        
        kldiv_foward[cls] = np.concatenate([kl_foward[cls][run] 
                                       for run in np.sort(kl_foward[cls].keys())])
        kldiv_reverse[cls] = np.concatenate([kl_reverse[cls][run] 
                                        for run in np.sort(kl_reverse[cls].keys())])
    
    return kldiv_foward, kldiv_reverse

from Functions.StatFunctions import KLDiv
window_range=30

k_fow, k_rev = kl_dv_fn(kdes, 1)
kl_foward_matrices = {
    'ClassA': np.zeros((window_range, k_fow['ClassA'].shape[0])),
    'ClassB': np.zeros((window_range, k_fow['ClassB'].shape[0])),
    'ClassC': np.zeros((window_range, k_fow['ClassC'].shape[0])),
    'ClassD': np.zeros((window_range, k_fow['ClassD'].shape[0]))
}
kl_reverse_matrices = {
    'ClassA': np.zeros((window_range, k_fow['ClassA'].shape[0])),
    'ClassB': np.zeros((window_range, k_fow['ClassB'].shape[0])),
    'ClassC': np.zeros((window_range, k_fow['ClassC'].shape[0])),
    'ClassD': np.zeros((window_range, k_fow['ClassD'].shape[0]))
}

for cls in k_fow:
        kl_foward_matrices[cls][0, :] = k_fow[cls]
        kl_reverse_matrices[cls][0, :] = k_rev[cls]
        
def get_kl(k):
    k_fow, k_rev = kl_dv_fn(kdes, k)
    for cls in k_fow:
        kl_foward_matrices[cls][k-1, :] = k_fow[cls]
        kl_reverse_matrices[cls][k-1, :] = k_fow[cls]
        
pool = Pool(4)
pool.map(get_kl, range(2,window_range+1))

foward_path = './foward_%i_%i_%i_%i.jbl' % (window, overlap, 
                                             decimation_rate, spectrum_bins_left)

reverse_path = './reverse_%i_%i_%i_%i.jbl' % (window, overlap, 
                                             decimation_rate, spectrum_bins_left)

joblib.dump(kl_foward_matrices, foward_path)
joblib.dump(kl_reverse_matrices, reverse_path)

