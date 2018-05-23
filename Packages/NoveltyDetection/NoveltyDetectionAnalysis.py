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
                 development_flag=False, development_events=400, model_prefix_str='RawData',verbose = True):
        
        m_time = time.time()
        
        # Analysis Characteristics
        self.analysis_name = analysis_name
        self.model_prefix_str = model_prefix_str
        
        # For multiprocessing purpose
        self.num_processes = multiprocessing.cpu_count()
        
        # Enviroment variables
        self.data_path = CONFIG['OUTPUTDATAPATH']
        self.results_path = CONFIG['PACKAGE_NAME']

        # paths to export results
        self.base_results_path = '%s/%s'%(self.results_path,self.analysis_name)
        self.pict_results_path = '%s/pictures_files'%(self.base_results_path)
        self.files_results_path = '%s/output_files'%(self.base_results_path)

        # Database caracteristics
        self.database = database
        self.n_pts_fft = n_pts_fft
        self.decimation_rate = decimation_rate
        self.spectrum_bins_left = spectrum_bins_left
        self.development_flag = development_flag
        self.development_events = development_events
        
        self.verbose = verbose
        
        # Check if LofarData has already been created...
        if not os.path.exists('%s/%s/lofar_data_file_fft_%i_decimation_%i_spectrum_left_%i.jbl'%(self.data_path,
                                                                                                 self.database,
                                                                                                 self.n_pts_fft,
                                                                                                 self.decimation_rate,
                                                                                                 self.spectrum_bins_left)):
            print 'No Files in %s/%s\n'%(self.data_path,
                                         self.database)
        else:
            #Read lofar data
            [data,trgt,class_labels] = joblib.load('%s/%s/lofar_data_file_fft_%i_decimation_%i_spectrum_left_%i.jbl'%(self.data_path,
                                                                                                 self.database,
                                                                                                 self.n_pts_fft,
                                                                                                 self.decimation_rate,
                                                                                                 self.spectrum_bins_left))


            m_time = time.time()-m_time
            print '[+] Time to read data file: '+str(m_time)+' seconds'

            # correct format
            self.all_data = data
            self.all_trgt = trgt
            self.trgt_sparse = np_utils.to_categorical(self.all_trgt.astype(int))

            # Process data
            # unbalanced data to balanced data with random data creation of small classes

            # Same number of events in each class
            self.qtd_events_biggest_class = 0
            self.biggest_class_label = ''
            
            # Define the class labels with ascii letters in uppercase
            self.class_labels = list(string.ascii_uppercase)[:self.trgt_sparse.shape[1]]

            for iclass, class_label in enumerate(self.class_labels):
                if sum(self.all_trgt==iclass) > self.qtd_events_biggest_class:
                    self.qtd_events_biggest_class = sum(self.all_trgt==iclass)
                    self.biggest_class_label = class_label
                if self.verbose:    
                    print "Qtd event of %s is %i"%(class_label,sum(self.all_trgt==iclass))
            if self.verbose:
                print "\nBiggest class is %s with %i events"%(self.biggest_class_label,self.qtd_events_biggest_class)

            self.balanceData()
            
            # turn targets in sparse mode
            self.trgt_sparse = np_utils.to_categorical(self.all_trgt.astype(int))
            
    def balanceData(self):
        # Balance data
        balanced_data = {}
        balanced_trgt = {}

        m_datahandler = dh.DataHandlerFunctions()

        for iclass, class_label in enumerate(self.class_labels):
            if self.development_flag:
                class_events = self.all_data[all_trgt==iclass,:]
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    



