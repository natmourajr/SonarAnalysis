#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This file contains the Novelty Detection Analysis with Stacked AutoEncoders
    Author: Vinicius dos Santos Mello <viniciusdsmello@poli.ufrj.br>
"""
import os
import sys
sys.path.insert(0,'..')

import pickle
import numpy as np
import time
import string
import multiprocessing

from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import metrics

from Functions import TrainParameters
from Functions import DataHandler as dh

from Functions.StackedAutoEncoders import StackedAutoEncoders
from Functions.StatisticalAnalysis import KLDiv, EstPDF

from NoveltyDetectionAnalysis import NoveltyDetectionAnalysis

num_processes = multiprocessing.cpu_count()

class SAENoveltyDetectionAnalysis(NoveltyDetectionAnalysis):
    
    def __init__(self, analysis_name='StackedAutoEncoder', database='4classes', n_pts_fft=1024, decimation_rate=3, spectrum_bins_left=400,
                 development_flag=False, development_events=400, model_prefix_str='RawData', n_folds=10, verbose=True): 
        
        super(SAENoveltyDetectionAnalysis, self).__init__(analysis_name, database, n_pts_fft, decimation_rate, spectrum_bins_left, development_flag,
                                                          development_events, model_prefix_str, verbose)

        self.analysis_name = analysis_name
        self.trn_params = None
        self.n_folds = n_folds
        self.CVO = None
    
    def setTrainParameters(self, n_inits=1, hidden_activation='tanh', output_activation='linear', n_epochs=300, n_folds=10, patience=30, batch_size=256,
                           verbose=False, optmizerAlgorithm='Adam', metrics=['accuracy'], loss='mean_squared_error'):
        self.trn_params_file='%s/%s/%s_trnparams.jbl'%(self.results_path,self.analysis_name,self.analysis_name)
        
        if not os.path.exists(self.trn_params_file):
            self.trn_params = TrainParameters.SAENoveltyDetectionTrnParams(n_inits=n_inits,
                                                                           hidden_activation=hidden_activation, # others tanh, relu, sigmoid, linear 
                                                                           output_activation=output_activation,
                                                                           n_epochs=n_epochs,
                                                                           patience=patience,
                                                                           batch_size=batch_size,
                                                                           verbose=verbose,
                                                                           optmizerAlgorithm=optmizerAlgorithm,
                                                                           metrics=metrics, #mean_squared_error
                                                                           loss=loss) #kullback_leibler_divergence
            self.trn_params.save(self.trn_params_file)
        else:
            self.trn_params = TrainParameters.SAENoveltyDetectionTrnParams()
            self.trn_params.load(self.trn_params_file)

        # Choose how many fold to be used in Cross Validation
        self.n_folds = n_folds
        self.CVO = TrainParameters.NoveltyDetectionFolds(folder=self.results_path,n_folds=self.n_folds,trgt=self.all_trgt,dev=self.development_flag,
                                                         verbose=True)
        print '\n'+self.trn_params.get_params_str()
    
    def getTrainParameters(self):
        self.setTrainParameters()
        return self.trn_params
    
    def clearTrainParametersFile(self):
        if os.path.exists(self.trn_params_folder):
            print "Removing Train Parameters file"
            os.remove(self.trn_params_file)
    
    
    def createSAEModels(self):
        self.SAE = {}
        self.trn_data = {}
        self.trn_trgt = {}
        self.trn_trgt_sparse = {}
        for inovelty in range(self.trgt_sparse.shape[1]):
            print "[*] Initializing SAE Class for class %s"%self.getClassLabels()[inovelty]
            self.trn_data[inovelty] = self.all_data[self.all_trgt!=inovelty]
            self.trn_trgt[inovelty] = self.all_trgt[self.all_trgt!=inovelty]
            self.trn_trgt[inovelty][self.trn_trgt[inovelty]>inovelty] = self.trn_trgt[inovelty][self.trn_trgt[inovelty]>inovelty]-1
            self.trn_trgt_sparse[inovelty] = np_utils.to_categorical(self.trn_trgt[inovelty].astype(int))
            # Initialize an SAE object for all novelties
            self.SAE[inovelty] = StackedAutoEncoders(params           = self.trn_params,
                                                     development_flag = self.development_flag,
                                                     n_folds          = self.n_folds,
                                                     save_path        = self.results_path,
                                                     CVO              = self.CVO,
                                                     noveltyDetection = True,
                                                     inovelty         = inovelty)

    
    def getSAEModels(self):
        return [self.SAE, self.trn_data, self.trn_trgt, self.trn_trgt_sparse]
    
    '''
        Method that implements different types of training through interfacing SAE methods
    '''
    def train(self, inovelty=0, fineTuning=False, trainingType="normal", data=None, trgt=None, ifold=0, hidden_neurons=[1], neurons_mat=[], layer=1,
              regularizer=None, regularizer_param=None, numThreads=num_processes):
        if fineTuning == False:
            fineTuning = 0
        else: 
            fineTuning = 1
        
        if regularizer != None and regularizer_param != None:
            sysCall = "python modelTrain.py --layer {0} --novelty {1} --finetunning {2} --threads {3} --regularizer {4} --paramvalue {5} --type {6}".format(
            layer, inovelty, fineTuning, numThreads, regularizer, regularizer_param, trainingType)
        else:
            sysCall = "python modelTrain.py --layer {0} --novelty {1} --finetunning {2} --threads {3} --type {4}".format(
            layer, inovelty, fineTuning, numThreads, trainingType)
        
        os.system(sysCall)
        

'''    def klDivergenceNeuronsVariation(self, inovelty = 0, layer = 1, hidden_neurons = range(400,0,-50),
                                     neurons_mat = [], regularizer=None, regularizer_param=None,
                                     clearPrevious = False, language='en'):
        # Neuron variation x KL Divergence
        neurons_mat = neurons_mat[:len(neurons_mat)-layer+2]

        # generate analysis data
        save_path=results_path

        current_analysis = 'klDivergence_%i_layer_%i_novelty'%(layer, inovelty)
        analysis_str = 'StackedAutoEncoder'
        model_prefix_str = 'RawData'

        analysis_file_name='%s/%s/%s_%s_neuron_number_sweep.jbl'%(results_path,analysis_str,analysis_name,current_analysis)

        # Plot parameters
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        plt.rcParams['legend.numpoints'] = 1
        plt.rcParams['legend.handlelength'] = 3
        plt.rcParams['legend.borderpad'] = 0.3
        plt.rcParams['legend.fontsize'] = 18
        m_colors = ['b', 'r', 'g', 'y']

        params_str = trn_params.get_params_str()

        klDivergenceFreq = {}
        klDivergenceKnown = np.zeros([self.n_folds, len(neurons_mat)], dtype=object)
        klDivergenceKnownFreq = {}
        klDivergenceNovelty = np.zeros([self.n_folds, len(neurons_mat)], dtype=object)
        klDivergenceNoveltyFreq = {}

        if clearPrevious:
            if os.path.exists(analysis_file_name):
                os.remove(analysis_file_name)

        if not os.path.exists(analysis_file_name):
            for ineuron in neurons_mat: 
                if ineuron == 0:
                    ineuron = 1
                neurons_str = SAE[inovelty].getNeuronsString(all_data, hidden_neurons=hidden_neurons[:layer-1]+[ineuron])

                models = {}
                outputs = {}
                norm_data = {}
                reconstructed_known_data = {}
                reconstructed_novelty_data = {}
                if verbose: 
                    print '[*] Layer: %i - Topology: %s'%(layer, neurons_str)

                n_bins = 100

                def getKlDiv(ifold):
                    train_id, test_id = CVO[inovelty][ifold]

                    # normalize known classes
                    if trn_params.params['norm'] == 'mapstd':
                        scaler = preprocessing.StandardScaler().fit(all_data[all_trgt!=inovelty][train_id,:])
                    elif trn_params.params['norm'] == 'mapstd_rob':
                        scaler = preprocessing.RobustScaler().fit(all_data[all_trgt!=inovelty][train_id,:])
                    elif trn_params.params['norm'] == 'mapminmax':
                        scaler = preprocessing.MinMaxScaler().fit(all_data[all_trgt!=inovelty][train_id,:])

                    known_data = scaler.transform(all_data[all_trgt!=inovelty][test_id,:])
                    novelty_data = scaler.transform(all_data[all_trgt==inovelty])

                    model = SAE[inovelty].getModel(data=all_data, trgt=all_trgt,
                                                   hidden_neurons=hidden_neurons[:layer-1]+[ineuron],
                                                   layer=layer, ifold=ifold)

                    known_output = model.predict(known_data)
                    novelty_output = model.predict(novelty_data)

                    klKnown = np.zeros([all_data.shape[1]], dtype=object)
                    klNovelty = np.zeros([all_data.shape[1]], dtype=object)

                    for ifrequency in range(0,400):
                        # Calculate KL Div for known data reconstruction
                        known_data_freq = known_data[:,ifrequency]
                        reconstructed_known_data = known_output[:,ifrequency]

                        m_bins = np.linspace(known_data_freq.min(), known_data_freq.max(), n_bins)

                        klKnown[ifrequency] = KLDiv(known_data_freq.reshape(-1,1), reconstructed_known_data.reshape(-1,1),
                                               bins=m_bins, mode='kernel', kernel='epanechnikov',
                                               kernel_bw=0.1, verbose=False)

                        klKnown[ifrequency] = klKnown[ifrequency][0]

                        # Calculate KL Div for novelty data reconstruction
                        novelty_data_freq = novelty_data[:,ifrequency]
                        reconstructed_novelty_data = novelty_output[:,ifrequency]

                        m_bins = np.linspace(novelty_data_freq.min(), novelty_data_freq.max(), n_bins)

                        klNovelty[ifrequency] = KLDiv(novelty_data_freq.reshape(-1,1), reconstructed_novelty_data.reshape(-1,1),
                                               bins=m_bins, mode='kernel', kernel='epanechnikov',
                                               kernel_bw=0.1, verbose=False)

                        klNovelty[ifrequency] = klNovelty[ifrequency][0]

                    return ifold, klKnown, klNovelty

                # Start Parallel processing
                p = multiprocessing.Pool(processes=num_processes)

                folds = range(len(CVO[inovelty]))
                if verbose:
                    print '[*] Calculating KL Div for all frequencies...'
                # Calculate the KL Div at all frequencies
                klDivergenceFreq[ineuron] = p.map(getKlDiv, folds)

                p.close()
                p.join()

                index = neurons_mat.index(ineuron)
                for ifold in range(n_folds):
                    klDivergenceKnownFreq = klDivergenceFreq[ineuron][ifold][1]
                    klDivergenceNoveltyFreq = klDivergenceFreq[ineuron][ifold][2]

                    klDivergenceKnown[ifold, index] = np.sum(klDivergenceKnownFreq)
                    klDivergenceNovelty[ifold, index] = np.sum(klDivergenceNoveltyFreq)

                joblib.dump([neurons_mat,klDivergenceKnown,klDivergenceNovelty],analysis_file_name,compress=9)
        else:
            [neurons_mat, klDivergenceKnown, klDivergenceNovelty] = joblib.load(analysis_file_name)
        
        # Plot results    
        fig, m_ax = plt.subplots(figsize=(20,30),nrows=5, ncols=2)

        for ifold in range(n_folds):
        irow = int(ifold/2)
        if (ifold % 2 == 0):
            icolumn = 0
        else: 
            icolumn = 1

        m_ax[irow, icolumn].plot(neurons_mat, klDivergenceKnown[ifold,:], 'b-o', label='Known Test Data')
        m_ax[irow, icolumn].plot(neurons_mat, klDivergenceNovelty[ifold,:], 'r--o', label='Novelty Data')
        m_ax[irow, icolumn].set_title('KL Divergence x Neurons - Layer %i - Fold %i - Novelty Class %i '%(layer,ifold+1, inovelty), fontsize=16,
                                      fontweight='bold')
        m_ax[irow, icolumn].set_ylabel('Kullback-Leibler Divergence', fontsize=22)
        m_ax[irow, icolumn].set_xlabel('Neurons', fontsize=22)
        m_ax[irow, icolumn].grid()
        m_ax[irow, icolumn].legend()
        plt.tight_layout()
        plt.show()
'''

    