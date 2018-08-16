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
from keras.models import load_model
from keras import backend as K

from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import metrics

from Functions.StackedAutoEncoders import StackedAutoEncoders

from NoveltyDetection import NoveltyDetectionAnalysis

num_processes = multiprocessing.cpu_count()


class SAENoveltyDetectionAnalysis(NoveltyDetectionAnalysis):

    def __init__(self, parameters=None, model_hash="", load_hash=False, verbose=False):
        super(SAENoveltyDetectionAnalysis, self).__init__(parameters, model_hash=model_hash, load_hash=load_hash,
                                                          verbose=False)

        self.SAE = {}
        self.trn_data = {}
        self.trn_trgt = {}
        self.trn_trgt_sparse = {}

    def createSAEModels(self):
        for inovelty in range(self.trgt_sparse.shape[1]):
            self.trn_data[inovelty] = self.all_data[self.all_trgt != inovelty]
            self.trn_trgt[inovelty] = self.all_trgt[self.all_trgt != inovelty]
            self.trn_trgt[inovelty][self.trn_trgt[inovelty] > inovelty] = self.trn_trgt[inovelty][
                                                                              self.trn_trgt[inovelty] > inovelty] - 1
            self.trn_trgt_sparse[inovelty] = np_utils.to_categorical(self.trn_trgt[inovelty].astype(int))

            # Initialize SAE objects for all novelties
            self.SAE[inovelty] = StackedAutoEncoders(params=self.parameters,
                                                     development_flag=self.development_flag,
                                                     n_folds=int(self.parameters['HyperParameters']['n_folds']),
                                                     save_path=self.getBaseResultsPath(),
                                                     CVO=self.CVO,
                                                     noveltyDetection=bool(self.parameters['NoveltyDetection']),
                                                     inovelty=inovelty
                                                     )

    def getSAEModels(self):
        return self.SAE

    def train(self, inovelty=0, fineTuning=False, trainingType="normal", ifold=0, hidden_neurons=[1],
              neurons_variation_step=50, layer=1, numThreads=num_processes):
        if fineTuning == False:
            fineTuning = 0
        else:
            fineTuning = 1

        hiddenNeuronsStr = str(hidden_neurons[0])
        if len(hidden_neurons) > 1:
            for ineuron in hidden_neurons[1:]:
                hiddenNeuronsStr = hiddenNeuronsStr + 'x' + str(ineuron)

        sysCall = "python modelTrain.py --layer {0} --novelty {1} --finetunning {2} --threads {3} --type {4} --hiddenNeurons {5} --neuronsVariationStep {6}".format(
            layer, inovelty, fineTuning, numThreads, trainingType, hiddenNeuronsStr, neurons_variation_step)

        if self.development_flag:
            sysCall = sysCall + " --developmentFlag 1 --developmentEvents " + str(self.development_events)

        print sysCall
        os.system(sysCall)
