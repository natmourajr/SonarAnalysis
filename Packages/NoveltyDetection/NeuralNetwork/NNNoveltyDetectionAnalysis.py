#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This file contains the Novelty Detection Analysis with Shallow Neural Networks
    Author: Vinicius dos Santos Mello <viniciusdsmello@poli.ufrj.br>
"""
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, '..')

import pickle
import numpy as np
import time
import string
import json
import multiprocessing

from keras.utils import np_utils
from sklearn.externals import joblib

from Functions.email_utils import EmailConnection, Email
from Functions.NeuralNetworks import NeuralNetworks

from NoveltyDetectionAnalysis import NoveltyDetectionAnalysis

from Functions.telegrambot import Bot

my_bot = Bot("lisa_thebot")

num_processes = multiprocessing.cpu_count()

class NNNoveltyDetectionAnalysis(NoveltyDetectionAnalysis):

    def __init__(self, parameters=None, model_hash=None, load_hash=False, load_data=True, verbose=False):
        super(NNNoveltyDetectionAnalysis, self).__init__(parameters=parameters, model_hash=model_hash, load_hash=load_hash, load_data=load_data, verbose=verbose)
        
        self.trn_data = {}
        self.trn_trgt = {}
        self.trn_trgt_sparse = {}
        self.models = {}
        
        for inovelty in range(self.all_trgt_sparse.shape[1]):
            self.trn_data[inovelty] = self.all_data[self.all_trgt != inovelty]
            self.trn_trgt[inovelty] = self.all_trgt[self.all_trgt != inovelty]
            self.trn_trgt[inovelty][self.trn_trgt[inovelty] > inovelty] = self.trn_trgt[inovelty][self.trn_trgt[inovelty] > inovelty] - 1
            self.trn_trgt_sparse[inovelty] = np_utils.to_categorical(self.trn_trgt[inovelty].astype(int))
            
            if self.parameters["HyperParameters"]["classifier_output_activation_function"] in ["tanh"]:
                self.trn_trgt_sparse[inovelty] = 2 * self.trn_trgt_sparse[inovelty] - np.ones(self.trn_trgt_sparse[inovelty].shape)
            
            self.models[inovelty] = NeuralNetworks(parameters=self.parameters,
                                                   save_path=self.getBaseResultsPath(),
                                                   CVO=self.CVO,
                                                   inovelty=inovelty, 
                                                   verbose=self.verbose
                                                   )
            
    def train(self, model_hash="", inovelty=0, trainingType="normal", ifold=0, hidden_neurons=[1],
              neurons_variation_step=50, layer=1, numThreads=num_processes):
        startTime = time.time()

        hiddenNeuronsStr = str(hidden_neurons[0])
        if len(hidden_neurons) > 1:
            for ineuron in hidden_neurons[1:]:
                hiddenNeuronsStr = hiddenNeuronsStr + 'x' + str(ineuron)

        sysCall = "python neuralnetwork_train.py --layer {0} --novelty {1} --threads {2} --type {3} --hiddenNeurons {4} --neuronsVariationStep {5} --modelhash {6}".format(
            layer, inovelty, numThreads, trainingType, hiddenNeuronsStr, neurons_variation_step, model_hash)
        print sysCall
        os.system(sysCall)
        
        duration = str(timedelta(seconds=float(time.time() - startTime)))
        message = "Technique: {}\n".format(self.parameters["Technique"])
        message = message + "Training Type: Neuron Sweep\n"
        message = message + "Novelty Class: {}\n".format(self.class_labels[inovelty])
        message = message + "Hash: {}\n".format(self.model_hash)
        message = message + "Duration: {}\n".format(duration)
        try:
            my_bot.sendMessage(message)
        except Exception as e:
            print("Erro ao enviar mensagem. Erro: " + str(e))
