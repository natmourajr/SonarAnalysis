# coding=utf-8
# Projeto Marinha do Brasil

# Author: Pedro Henrique Braga Lisboa
# Laboratorio de Processamento de Sinais - UFRJ
# Laboratório de Tecnologia Sonar - UFRJ/Marinha do Brasil

from __future__ import division, print_function

import os
import sys

import joblib

from Functions.SonarFunctions.read_raw_data import AudioData
from Functions.SonarFunctions.lofar_analysis import LofarAnalysis, tpsw
from Functions.DataHandler import LofarDataset
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, decimate

import numpy as np
datapath = os.getenv('OUTPUTDATAPATH')
audiodatapath = os.getenv('INPUTDATAPATH')
results_path = os.getenv('PACKAGE_NAME')
database = '4classes'


# TODO check for errors in user entry
n_pts_fft = sys.argv[1]
n_overlap = sys.argv[2]
decimation_rate = sys.argv[3]
spectrum_bins_left = sys.argv[4]
# TODO receive verbose
verbose=1
if verbose:
    print("Performing Lofar Analysis in %s database" % database)

ad = AudioData(audiodatapath, database)
raw_data,fs = ad.read_raw_data(verbose=verbose)

la = LofarAnalysis(decimation_rate, n_pts_fft, n_overlap, spectrum_bins_left)
lofar_data = la.from_raw_data(raw_data, fs, database, datapath)

#joblib.dump


