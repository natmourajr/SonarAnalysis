#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file contains the Novelty Detection Analysis Base Class
    Author: Vinicius dos Santos Mello <viniciusdsmello@poli.ufrj.br>
"""
import os
import sys
sys.path.insert(0,'..')

from noveltyDetectionConfig import CONFIG

import time
import string
import multiprocessing
import numpy as np

from sklearn.externals import joblib

from keras.utils import np_utils

from Functions import TrainParameters as trnparams
from Functions import DataHandler as dh

class NoveltyDetectionAnalysis(object):
    def __init__(self, analysis_name='', database='4classes', n_pts_fft=1024, decimation_rate=3, spectrum_bins_left=400,
                 n_windows = 1, development_flag=False, development_events=400, model_prefix_str='RawData',verbose = True, loadData = True):
        
        # Analysis Characteristics
        self.analysis_name = analysis_name
        self.model_prefix_str = model_prefix_str
        
        # For multiprocessing purpose
        self.num_processes = multiprocessing.cpu_count()
        
        # Enviroment variables
        self.DATA_PATH = CONFIG['OUTPUTDATAPATH']
        self.RESULTS_PATH = CONFIG['PACKAGE_NAME']

        # paths to export results
        self.base_results_path = os.path.join(self.RESULTS_PATH,self.analysis_name)
        
        # Database caracteristics
        self.database = database
        self.n_pts_fft = n_pts_fft
        self.decimation_rate = decimation_rate
        self.spectrum_bins_left = spectrum_bins_left
        self.development_flag = development_flag
        self.development_events = development_events
        
        self.verbose = verbose
        
        # Set the number of windows to be used per event
        self.n_windows = n_windows
        
        if(loadData):
            self.loadData()
        
        
    def loadData(self):
        m_time = time.time()
        # Check if LofarData has already been created...
        data_file = os.path.join(self.DATA_PATH,self.database,
                                 "lofar_data_file_fft_%i_decimation_%i_spectrum_left_%i.jbl"%(self.n_pts_fft,
                                                                                              self.decimation_rate,
                                                                                              self.spectrum_bins_left
                                                                                             )
                                )
        
        if not os.path.exists(data_file):
            print 'No Files in %s/%s\n'%(self.DATA_PATH, self.database)
        else:
            #Read lofar data
            [self.all_data,self.all_trgt,class_labels] = joblib.load(data_file)


            m_time = time.time()-m_time
            print '[+] Time to read data file: '+str(m_time)+' seconds'

            # correct format
            self.trgt_sparse = np_utils.to_categorical(self.all_trgt.astype(int))

            # Process data
            # unbalanced data to balanced data with random data creation of small classes

            # Same number of events in each class
            self.qtd_events_biggest_class = 0
            self.biggest_class_label = ''
            
            # Define the class labels with ascii letters in uppercase
            self.class_labels = list(string.ascii_uppercase)[:self.trgt_sparse.shape[1]]

            new_target = np.zeros(0)
            new_data = np.zeros([0,self.all_data.shape[1]*self.n_windows])

            for iclass, label in enumerate(self.class_labels):
                data = self.all_data[self.all_trgt==iclass]
                data = np.reshape(data, [data.shape[0]/self.n_windows, data.shape[1]*self.n_windows])
                
                trgt = np.ones(data.shape[0]) * iclass
                new_data = np.concatenate((new_data, data), axis=0)
                new_target = np.concatenate((new_target, trgt), axis=0)
            
            self.all_data = new_data
            self.all_trgt = new_target
            
            for iclass, class_label in enumerate(self.class_labels):
                if sum(self.all_trgt==iclass) > self.qtd_events_biggest_class:
                    self.qtd_events_biggest_class = sum(self.all_trgt==iclass)
                    self.biggest_class_label = class_label
                if self.verbose:    
                    print "Qtd event of %s is %i"%(class_label,sum(self.all_trgt==iclass))
            if self.verbose:
                print "\nBiggest class is %s with %i events"%(self.biggest_class_label,self.qtd_events_biggest_class)

                        
                
            # self.balanceData()
            
            
            # turn targets in sparse mode
            self.trgt_sparse = np_utils.to_categorical(self.all_trgt.astype(int))
            
    def balanceData(self):
        # Balance data
        balanced_data = {}
        balanced_trgt = {}

        m_datahandler = dh.DataHandlerFunctions()

        for iclass, class_label in enumerate(self.class_labels):
            if self.development_flag:
                class_events = self.all_data[self.all_trgt==iclass,:]
                if len(balanced_data) == 0:
                    balanced_data = class_events[0:self.development_events,:]
                    balanced_trgt = (iclass)*np.ones(self.development_events)
                else:
                    balanced_data = np.append(balanced_data,class_events[0:self.development_events,:],axis=0)
                    balanced_trgt = np.append(balanced_trgt,(iclass)*np.ones(self.development_events))
            else:
                if len(balanced_data) == 0:
                    class_events = self.all_data[self.all_trgt==iclass,:]
                    balanced_data = m_datahandler.CreateEventsForClass(class_events,self.qtd_events_biggest_class-(len(class_events)))
                    balanced_trgt = (iclass)*np.ones(self.qtd_events_biggest_class)
                else:
                    class_events = self.all_data[self.all_trgt==iclass,:]
                    created_events = (m_datahandler.CreateEventsForClass(self.all_data[self.all_trgt==iclass,:], 
                                                                         self.qtd_events_biggest_class-(len(class_events))))
                    balanced_data = np.append(balanced_data,created_events,axis=0)
                    balanced_trgt = np.append(balanced_trgt, (iclass)*np.ones(created_events.shape[0]),axis=0)

        self.all_data = balanced_data
        self.all_trgt = balanced_trgt
        
    def getData(self):
        return [self.all_data, self.all_trgt, self.trgt_sparse]

    def getClassLabels(self):
        return self.class_labels
    
    def setTrainParameters(self):
        pass
    
    def clearTrainParametersFile(self):
        pass
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    



