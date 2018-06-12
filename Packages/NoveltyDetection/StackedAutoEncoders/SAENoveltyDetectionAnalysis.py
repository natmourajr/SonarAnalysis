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

from noveltyDetectionConfig import CONFIG
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
        self.baseResultsPath = ''
        
        self.development_flag = development_flag
        self.development_events = development_events
    
    def setTrainParameters(self, n_inits=1, hidden_activation='tanh', output_activation='linear', classifier_output_activation = 'softmax', n_epochs=300,
                           n_folds=10, patience=30, batch_size=256, verbose=False, optmizerAlgorithm='Adam', metrics=['accuracy'], loss='mean_squared_error'):        
        self.trn_params = TrainParameters.SAENoveltyDetectionTrnParams(n_inits=n_inits,
                                                                       folds=n_folds,
                                                                       hidden_activation=hidden_activation, # others tanh, relu, sigmoid, linear 
                                                                       output_activation=output_activation,
                                                                       classifier_output_activation=classifier_output_activation,
                                                                       n_epochs=n_epochs,
                                                                       patience=patience,
                                                                       batch_size=batch_size,
                                                                       verbose=verbose,
                                                                       optmizerAlgorithm=optmizerAlgorithm,
                                                                       metrics=metrics, #mean_squared_error
                                                                       loss=loss) #kullback_leibler_divergence
        
        self.modelPath = self.trn_params.getModelPath()
        self.baseResultsPath = self.getBaseResultsPath()

        self.trn_params_file = os.path.join(self.baseResultsPath, "trnparams.jbl")
        
        if not os.path.exists(self.trn_params_file):
            self.trn_params.save(self.trn_params_file)
            self.trn_params.save(os.path.join(self.RESULTS_PATH,self.analysis_name, "trnparams.jbl"))
        
        # Choose how many fold to be used in Cross Validation
        self.CVO = TrainParameters.NoveltyDetectionFolds(folder=self.RESULTS_PATH,n_folds=self.n_folds,trgt=self.all_trgt,dev=self.development_flag,
                                                         verbose=True)
        
    def getBaseResultsPath(self):
        self.modelPath = self.trn_params.getModelPath()
        self.baseResultsPath = os.path.join(self.RESULTS_PATH, self.modelPath)
        
        if not os.path.exists(self.baseResultsPath):
            print ("Creating " + self.baseResultsPath)
            os.makedirs(self.baseResultsPath)
        
        return self.baseResultsPath
        
    def getTrainParameters(self):
        self.setTrainParameters()
        return self.trn_params
                                 
    def loadTrainParameters(self):
        params_file = os.path.join(self.RESULTS_PATH, self.analysis_name, "trnparams.jbl")
        if (os.path.exists(params_file)):
            self.trn_params = TrainParameters.SAENoveltyDetectionTrnParams()
            self.trn_params.load(params_file)
            # Choose how many fold to be used in Cross Validation
            self.CVO = TrainParameters.NoveltyDetectionFolds(folder=self.RESULTS_PATH,n_folds=self.n_folds,trgt=self.all_trgt,dev=self.development_flag,
                                                             verbose=True)
        else: 
            self.setTrainParameters()
            
    
    def clearTrainParametersFile(self):
        if os.path.exists(self.trn_params_folder):
            print "Removing Train Parameters file"
            os.remove(self.trn_params_file)
        
    
    def createSAEModels(self):
        self.SAE = {}
        self.trn_data = {}
        self.trn_trgt = {}
        self.trn_trgt_sparse = {}
        if(self.trn_params == None):
            self.loadTrainParameters()
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
                                                     save_path        = self.getBaseResultsPath(),
                                                     CVO              = self.CVO,
                                                     noveltyDetection = True,
                                                     inovelty         = inovelty)

    
    def getSAEModels(self):
        return [self.SAE, self.trn_data, self.trn_trgt, self.trn_trgt_sparse]
    
    '''
        Method that implements different types of training through interfacing SAE methods
    '''
    def train(self, inovelty=0, fineTuning=False, trainingType="normal", ifold=0, hidden_neurons=[1], neurons_variation_step=50, layer=1,
              regularizer=None, regularizer_param=None, numThreads=num_processes):
        if fineTuning == False:
            fineTuning = 0
        else: 
            fineTuning = 1
            
        hiddenNeuronsStr = str(hidden_neurons[0])
        if len(hidden_neurons) > 1:
            for ineuron in hidden_neurons[1:]:
                hiddenNeuronsStr = hiddenNeuronsStr + 'x' + str(ineuron)
        
        if regularizer != None and regularizer_param != None:
            sysCall = "python modelTrain.py --layer {0} --novelty {1} --finetunning {2} --threads {3} --regularizer {4} --paramvalue {5} --type {6} --hiddenNeurons {7} --neuronsVariationStep {8}".format(
            layer, inovelty, fineTuning, numThreads, regularizer, regularizer_param, trainingType, hiddenNeuronsStr, neurons_variation_step)
        else:
            sysCall = "python modelTrain.py --layer {0} --novelty {1} --finetunning {2} --threads {3} --type {4} --hiddenNeurons {5} --neuronsVariationStep {6}".format(layer, inovelty, fineTuning, numThreads, trainingType, hiddenNeuronsStr, neurons_variation_step)
        
        if self.development_flag:
            sysCall = sysCall + " --developmentFlag 1 --developmentEvents " + str(self.development_events)
                                    
        print sysCall
        os.system(sysCall)
        