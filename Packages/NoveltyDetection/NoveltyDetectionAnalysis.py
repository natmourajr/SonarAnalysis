#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file contains the Novelty Detection Analysis Base Class
    Author: Vinicius dos Santos Mello <viniciusdsmello@poli.ufrj.br>
"""
import os
import sys
import shutil

sys.path.insert(0, '..')

import noveltyDetectionConfig

import time
import string
import json
import hashlib
import multiprocessing
import numpy as np
from sklearn.externals import joblib
from sklearn import metrics
from sklearn import model_selection
import matplotlib.pyplot as plt
from keras.utils import np_utils
from Functions import TrainParameters
from Functions import DataHandler as dh

class NoveltyDetectionAnalysis(object):
    def __init__(self, parameters=None, model_hash=None, load_hash=False, load_data=True, verbose=True):
        
        self.all_data = None
        self.all_trgt = None
        self.all_trgt_sparse = None
        self.class_labels = None
        self.verbose = verbose
        
        # Enviroment variables
        self.DATA_PATH = noveltyDetectionConfig.CONFIG['OUTPUTDATAPATH']
        self.RESULTS_PATH = noveltyDetectionConfig.CONFIG['PACKAGE_NAME']

        # paths to export results
        
        if load_hash and model_hash is not None:
            self.model_hash = model_hash
            self.baseResultsPath = os.path.join(self.RESULTS_PATH, parameters["Technique"], "outputs", self.model_hash)
            self.parameters_file = os.path.join(self.baseResultsPath, "parameters.json")
            self.loadTrainParametersByHash(model_hash)
        else:
            if (parameters == None):
                print("Parameters must not be None!")
                exit()
            self.parameters = parameters
            # Set the hash of the JSON text with the parameters
            self.model_hash = hashlib.sha256(json.dumps(parameters)).hexdigest()
            self.baseResultsPath = os.path.join(self.RESULTS_PATH, self.parameters["Technique"], "outputs", self.model_hash)
            self.parameters_file = os.path.join(self.baseResultsPath, "parameters.json")
        
        self.analysis_output_folder = os.path.join(self.baseResultsPath, "AnalysisFiles")
        self.pictures_output_folder = os.path.join(self.baseResultsPath, "Pictures")
        
        if not os.path.exists(self.baseResultsPath):
            print ("Creating " + self.baseResultsPath)
            os.makedirs(self.baseResultsPath)
            
        if not os.path.exists(self.analysis_output_folder):
            print ("Creating " + self.analysis_output_folder)
            os.makedirs(self.analysis_output_folder)
            
        if not os.path.exists(self.pictures_output_folder):
            print ("Creating " + self.pictures_output_folder)
            os.makedirs(self.pictures_output_folder)
            
        # Save parameters file
        if not os.path.exists(self.parameters_file):
            with open(self.parameters_file, "w") as f:
                print ("Saving " + self.parameters_file)
                json.dump(self.parameters, f)
            
        # Database caracteristics
        self.database = self.parameters['InputDataConfig']['database']
        self.n_pts_fft = int(self.parameters['InputDataConfig']['n_pts_fft'])
        self.decimation_rate = int(self.parameters['InputDataConfig']['decimation_rate'])
        self.spectrum_bins_left = int(self.parameters['InputDataConfig']['spectrum_bins_left'])

        # Set the number of time windows to be used per event
        self.n_windows = int(self.parameters['InputDataConfig']['n_windows'])

        self.development_flag = bool(self.parameters['DevelopmentMode'])
        self.development_events = int(self.parameters['DevelopmentEvents'])

        self.n_folds = self.parameters["HyperParameters"]["n_folds"]
        
        if load_data:
            self.loadData()

        self.CVO = self.getCVO()

        # For multiprocessing purpose
        self.num_processes = multiprocessing.cpu_count()

    def loadData(self):
        m_time = time.time()
        # Check if LofarData has already been created...
        data_file = os.path.join(self.DATA_PATH, self.database,
                                 "lofar_data_file_fft_%i_decimation_%i_spectrum_left_%i.jbl" % (self.n_pts_fft,
                                                                                                self.decimation_rate,
                                                                                                self.spectrum_bins_left
                                                                                                )
                                 )

        if not os.path.exists(data_file):
            print 'No Files in %s/%s\n' % (self.DATA_PATH, self.database)
        else:
            # Read lofar data
            [self.all_data, self.all_trgt, class_labels] = joblib.load(data_file)

            m_time = time.time() - m_time
            print '[+] Time to read data file: ' + str(m_time) + ' seconds'

            # correct format
            self.all_trgt_sparse = np_utils.to_categorical(self.all_trgt.astype(int))

            # Same number of events in each class
            self.qtd_events_biggest_class = 0
            self.biggest_class_label = ''

            # Define the class labels with ascii letters in uppercase
            self.class_labels = list(string.ascii_uppercase)[:self.all_trgt_sparse.shape[1]]

            new_target = np.zeros(0)
            new_data = np.zeros([0, self.all_data.shape[1] * self.n_windows])

            for iclass, label in enumerate(self.class_labels):
                data = self.all_data[self.all_trgt == iclass]
                data = np.reshape(data, [data.shape[0] / self.n_windows, data.shape[1] * self.n_windows])

                trgt = np.ones(data.shape[0]) * iclass
                new_data = np.concatenate((new_data, data), axis=0)
                new_target = np.concatenate((new_target, trgt), axis=0)

            self.all_data = new_data
            self.all_trgt = new_target

            for iclass, class_label in enumerate(self.class_labels):
                if sum(self.all_trgt == iclass) > self.qtd_events_biggest_class:
                    self.qtd_events_biggest_class = sum(self.all_trgt == iclass)
                    self.biggest_class_label = class_label
                if self.verbose:
                    print "Qtd event of {} is {:d}".format(class_label, sum(self.all_trgt == iclass))
            if self.verbose:
                print "\nBiggest class is {} with {:d} events".format(self.biggest_class_label,
                                                                      self.qtd_events_biggest_class)
                print "Total of events in the dataset is {:d}".format(self.all_trgt.shape[0])

            if bool(self.parameters["InputDataConfig"]["balance_data"]):
                print "Balacing data..."
                self.balanceData()

            # turn targets in sparse mode
            self.all_trgt_sparse = np_utils.to_categorical(self.all_trgt.astype(int))
            
            if self.parameters["HyperParameters"]["classifier_output_activation_function"] in ["tanh"]:
                # Transform the output into [-1,1]
                self.all_trgt_sparse = 2*self.all_trgt_sparse-np.ones(self.all_trgt_sparse.shape)

    def balanceData(self):
        # Process data
        # unbalanced data to balanced data with random data creation of small classes

        # Balance data
        balanced_data = {}
        balanced_trgt = {}

        m_datahandler = dh.DataHandlerFunctions()

        for iclass, class_label in enumerate(self.class_labels):
            if self.development_flag:
                class_events = self.all_data[self.all_trgt == iclass, :]
                if len(balanced_data) == 0:
                    balanced_data = class_events[0:self.development_events, :]
                    balanced_trgt = (iclass) * np.ones(self.development_events)
                else:
                    balanced_data = np.append(balanced_data, class_events[0:self.development_events, :], axis=0)
                    balanced_trgt = np.append(balanced_trgt, (iclass) * np.ones(self.development_events))
            else:
                if len(balanced_data) == 0:
                    class_events = self.all_data[self.all_trgt == iclass, :]
                    balanced_data = m_datahandler.CreateEventsForClass(class_events, self.qtd_events_biggest_class - (
                        len(class_events)))
                    balanced_trgt = (iclass) * np.ones(self.qtd_events_biggest_class)
                else:
                    class_events = self.all_data[self.all_trgt == iclass, :]
                    created_events = (m_datahandler.CreateEventsForClass(self.all_data[self.all_trgt == iclass, :],
                                                                         self.qtd_events_biggest_class - (
                                                                             len(class_events))))
                    balanced_data = np.append(balanced_data, created_events, axis=0)
                    balanced_trgt = np.append(balanced_trgt, (iclass) * np.ones(created_events.shape[0]), axis=0)

        self.all_data = balanced_data
        self.all_trgt = balanced_trgt

    def getData(self):
        return [self.all_data, self.all_trgt, self.all_trgt_sparse]

    def getClassLabels(self):
        return self.class_labels

    def getCVO(self):
        # Cross Validation
        n_folds = self.parameters["HyperParameters"]["n_folds"]
        if n_folds < 2:
            print 'Invalid number of folds'
            return -1
        if bool(self.parameters["InputDataConfig"]["balance_data"]):
            file_name = os.path.join(self.RESULTS_PATH, "%i_folds_cross_validation_balanced_data.jbl"%(n_folds))
        else: 
            file_name = os.path.join(self.RESULTS_PATH, "%i_folds_cross_validation.jbl"%(n_folds))

        if not os.path.exists(file_name):
            if self.verbose:
                print "Creating %s"%(file_name)

            if self.all_trgt is None:
                print 'Invalid trgt'
                return -1

            CVO = {}
            for inovelty, novelty_class in enumerate(np.unique(self.all_trgt)):
                skf = model_selection.StratifiedKFold(n_splits=n_folds)
                process_trgt = self.all_trgt[self.all_trgt!=novelty_class]
                CVO[inovelty] = skf.split(X = np.zeros(process_trgt.shape), y=process_trgt)
                CVO[inovelty] = list(CVO[inovelty])
            if self.verbose:
                print 'Saving in %s'%(file_name)

            joblib.dump([CVO],file_name,compress=9)
        else:
            if self.verbose:
                print "Reading from %s"%(file_name)

            [CVO] = joblib.load(file_name)
        return CVO

    def getBaseResultsPath(self):
        return self.baseResultsPath

    def getTrainParameters(self):
        return self.parameters

    def loadTrainParametersByHash(self, model_hash):
        # Get JSON text from Database
        # TODO...
        json_str = ""
        
        if os.path.exists(self.parameters_file):
            with open(self.parameters_file, "r") as file:
                print("Reading from " + self.parameters_file)
                self.parameters = json.load(file)
    
    def plot_train_history_loss(self, history):
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()

    def relative_auc(self, x, y, xlim = [0,1], ylim=[0,1]):
        total_area = (xlim[1] - xlim[0])*(ylim[1]-ylim[0])
        rel_area = metrics.auc(x, y)/total_area
        return rel_area
    
    def get_results_zip(self):
        shutil.make_archive(self.baseResultsPath, 'zip', self.baseResultsPath)
        return self.baseResultsPath + ".zip"

    def get_pictures_zip(self):
        shutil.make_archive(self.pictures_output_folder, 'zip', self.pictures_output_folder)
        return self.pictures_output_folder + ".zip"

    def get_analysis_zip(self):
        shutil.make_archive(self.analysis_output_folder, 'zip', self.analysis_output_folder)
        return self.analysis_output_folder + ".zip"