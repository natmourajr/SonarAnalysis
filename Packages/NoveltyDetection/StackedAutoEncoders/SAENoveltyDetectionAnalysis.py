#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This file contains the Novelty Detection Analysis with Stacked AutoEncoders
    Author: Vinicius dos Santos Mello <viniciusdsmello@poli.ufrj.br>
"""
import os
import sys

sys.path.insert(0, '..')

import pickle
import numpy as np
import time
import string
import json
import multiprocessing

from keras.utils import np_utils
from sklearn.externals import joblib
from sklearn import preprocessing

from Functions.StackedAutoEncoders import StackedAutoEncoders

from NoveltyDetectionAnalysis import NoveltyDetectionAnalysis

num_processes = multiprocessing.cpu_count()


class SAENoveltyDetectionAnalysis(NoveltyDetectionAnalysis):

    def __init__(self, parameters=None, model_hash=None, load_hash=False, load_data=True, verbose=False):
        super(SAENoveltyDetectionAnalysis, self).__init__(parameters=parameters, model_hash=model_hash, load_hash=load_hash, load_data=load_data, verbose=verbose)
        
        self.trn_data = {}
        self.trn_trgt = {}
        self.trn_trgt_sparse = {}
        self.SAE = {}
        
        for inovelty in range(self.all_trgt_sparse.shape[1]):
            self.trn_data[inovelty] = self.all_data[self.all_trgt != inovelty]
            self.trn_trgt[inovelty] = self.all_trgt[self.all_trgt != inovelty]
            self.trn_trgt[inovelty][self.trn_trgt[inovelty] > inovelty] = self.trn_trgt[inovelty][self.trn_trgt[inovelty] > inovelty] - 1
            self.trn_trgt_sparse[inovelty] = np_utils.to_categorical(self.trn_trgt[inovelty].astype(int))
            
            if self.parameters["HyperParameters"]["classifier_output_activation_function"] in ["tanh"]:
                self.trn_trgt_sparse[inovelty] = 2 * self.trn_trgt_sparse[inovelty] - np.ones(self.trn_trgt_sparse[inovelty].shape)
            
            
    def createSAEModels(self):        
        for inovelty in range(self.all_trgt_sparse.shape[1]):
            # Initialize SAE objects for all novelties
            self.SAE[inovelty] = StackedAutoEncoders(parameters=self.parameters,
                                                     save_path=self.getBaseResultsPath(),
                                                     CVO=self.CVO,
                                                     inovelty=inovelty, 
                                                     verbose=self.verbose
                                                     )

        return self.SAE

    def train(self, model_hash="", inovelty=0, fineTuning=False, trainingType="normal", ifold=0, hidden_neurons=[1],
              neurons_variation_step=50, layer=1, numThreads=num_processes):
        if fineTuning == False:
            fineTuning = 0
        else:
            fineTuning = 1

        hiddenNeuronsStr = str(hidden_neurons[0])
        if len(hidden_neurons) > 1:
            for ineuron in hidden_neurons[1:]:
                hiddenNeuronsStr = hiddenNeuronsStr + 'x' + str(ineuron)

        sysCall = "python sae_train.py --layer {0} --novelty {1} --finetunning {2} --threads {3} --type {4} --hiddenNeurons {5} --neuronsVariationStep {6} --modelhash {7}".format(
            layer, inovelty, fineTuning, numThreads, trainingType, hiddenNeuronsStr, neurons_variation_step, model_hash)
        print sysCall
        os.system(sysCall)

    def get_data_scaler(self, inovelty=0, ifold=0):
        train_id, test_id = self.CVO[inovelty][ifold]

        # normalize known classes
        if self.parameters["HyperParameters"]["norm"] == "mapstd":
            scaler = preprocessing.StandardScaler().fit(self.all_data[self.all_trgt!=inovelty][train_id,:])
        elif self.parameters["HyperParameters"]["norm"] == "mapstd_rob":
            scaler = preprocessing.RobustScaler().fit(self.all_data[self.all_trgt!=inovelty][train_id,:])
        elif self.parameters["HyperParameters"]["norm"] == "mapminmax":
            scaler = preprocessing.MinMaxScaler().fit(self.all_data[self.all_trgt!=inovelty][train_id,:])
        
        return scaler