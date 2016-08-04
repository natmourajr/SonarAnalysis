# Projeto Marinha do Brasil

# Autor: Natanael Junior (natmourajr@gmail.com)
# Laboratorio de Processamento de Sinais - UFRJ
# Laboratorio de Tecnologia Sonar - UFRJ/Marinha do Brasil


import sys
import os
import os.path

import wave
import pickle

from sklearn.externals import joblib

import numpy as np

from scipy.io import loadmat
from sklearn.externals import joblib

print 'Starting '+os.path.basename(__file__)

# System var. point to external folders
# In vew version there is only 1 file in outputdatapath
inputpath = os.environ['INPUTDATAPATH']
outputpath = os.environ['OUTPUTDATAPATH']

# check if a file exists

# Variable to chance Database
subfolder = "4classes"

# Hard Codded?! 
n_pts_fft = 1024
decimation_rate = 3

if os.path.exists("%s/LofarData_%s_%i_fft_pts_%i_decimation_rate.mat"%(outputpath,subfolder,n_pts_fft,decimation_rate)):
    print "Processing File: "+"LofarData_%s_%i_fft_pts_%i_decimation_rate.mat"%(subfolder,n_pts_fft,decimation_rate)
    matfile = loadmat("%s/LofarData_%s_%i_fft_pts_%i_decimation_rate.mat"%(outputpath,subfolder,n_pts_fft,decimation_rate))

    # get a list of elements in inputdatapath
    for dirname, dirnames, filenames in os.walk('%s/%s'%(inputpath,subfolder),topdown=False):
        dirnames.sort() # to be in correct order
    ships = dirnames

    data = {}
    for iship, ship in enumerate(ships):
        print "Processing ship: "+ship
        data[iship] = {}
        for irun in range(matfile['data_lofar'][0,0][ship][0,0]['run'].shape[1]):
            print 'Processing Run: '+str(irun)
            data[iship][irun] = {}
            run = {}
            run['Fs'] = matfile['Fs'][0,0]
            [F, T] = matfile['data_lofar'][0,0][ship][0,0]['run'][0,irun].shape
            run['Freq'] = np.linspace(0, run['Fs']/2.0, F)
            run['Time'] = np.linspace(0, T * 1./run['Fs'], T)
            run['Signal'] = matfile['data_lofar'][0,0][ship][0,0]['run'][0,irun]
	    run['Windows'] = matfile['data_lofar'][0,0][ship][0,0]['windonazed_run'][0,irun]
            
            # add signal
            data[iship][irun] = run
    class_labels = ships
    
    #file = "%s/LofarData_%s_%i_fft_pts_%i_decimation_rate.jbl"%(outputpath,subfolder,n_pts_fft,decimation_rate)
    #joblib.dump([data,class_labels],file,compress=9)

    joblib.dump([data,class_labels],"%s/LofarData_%s_%i_fft_pts_%i_decimation_rate.jbl"%(outputpath,subfolder,n_pts_fft,decimation_rate),compress=9)



