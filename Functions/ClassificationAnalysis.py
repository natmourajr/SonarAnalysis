"""
    This file contents some classification analysis functions
"""

import os
import time
import numpy as np

from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn import preprocessing



class TrnInformation(object):
    def __init__(self, date='', n_folds=2, n_inits=2,
                 norm='mapstd',verbose=False,
                 train_verbose = False):
        self.n_folds = n_folds
        self.n_inits = n_inits
        self.norm = norm
        self.verbose = verbose
        self.train_verbose = train_verbose
        
        self.CVO = None
        if date == '':
            self.date = time.strftime("%Y_%m_%d_%H_%M_%S")
        else:
            self.date = date
    
    def Print(self):
        print 'Class TrnInformation'
        print '\tDate %s'%(self.date)
        print '\tNumber of Folds %i'%(self.n_folds)
        print '\tNumber of Initializations: %i'%(self.n_inits)
        print '\tNormalization: %s'%(self.norm)
        if self.CVO is None:
            print '\tCVO is None'
        else:
            print '\tCVO is ', self.CVO

    def SplitTrainSet(self,trgt):
        # divide data in train and test for novelty detection
        CVO = cross_validation.StratifiedKFold(trgt, self.n_folds)
        self.CVO = list(CVO)

    def save(self, path=''):
        print 'Save TrnInformation'
        if path == '':
            print 'No valid path...'
            return -1
        joblib.dump([self.date,self.n_folds, self.n_inits, self.norm, self.CVO],path,compress=9)
        return 0

    def load(self, path):
        print 'Load TrnInformation'
        if not os.path.exists(path):
            print 'No valid path...'
            return -1
        [self.date,self.n_folds, self.n_inits, self.norm, self.CVO] = joblib.load(path)


class ClassificationBaseClass(object):
    
    name = 'BaseClass'
    preproc_path = ''
    train_path = ''
    anal_path = ''
        
    date = None
    trn_info = None
    
    preproc_done = False
    training_done = False
    analysis_done = False
    
    def __init__(self, name='BaseClass', preproc_path='', train_path='', anal_path=''):
        self.name = name
        self.preproc_path = preproc_path
        self.train_path = train_path
        self.anal_path = anal_path
        
        self.date = None
        self.trn_info = None

        self.preproc_done = False
        self.training_done = False
        self.analysis_done = False

    def Print(self):
        print 'Class %s'%(self.name)
        print '\tPre-Proc. Data Path: ', self.preproc_path
        print '\tTraining Data Path: ', self.train_path
        print '\tAnalysis Data Path: ', self.anal_path

        if self.preproc_done:
            print '\tPreProcessing was done'
        else:
            print '\tPreProcessing was not done'

        if self.training_done:
            print '\tTraining Proc. was done'
        else:
            print '\tTraining Proc. was not done'

        if self.analysis_done:
            print '\tAnalysis Proc. was done'
        else:
            print '\tAnalysis Proc. was not done'

    def preprocess(self, data, trgt, trn_info=None):
        print 'ClassificationAnalysis preprocess function'
        if self.trn_info is None:
            if trn_info is None:
                self.trn_info = TrnInformation()
            else:
                self.trn_info = trn_info

        # preprocess
        self.date = date

        # save preprocess
        self.preproc_done = True

    def train(self, data, trgt):
        print 'ClassificationAnalysis train function'
        if not self.preproc_done:
            self.preprocess(data,trgt)
        
        # train process
        
        self.training_done = True

    def analysis(self, data, trgt):
        print 'ClassificationAnalysis analysis function'
        if not self.training_done:
            self.train(data,trgt)
        
        # analysis process

        self.analysis_done = True

class NeuralClassification(ClassificationBaseClass):
    def preprocess(self, data, trgt, trn_info=None,fold=0):
        print 'NeuralClassication preprocess function'
        if self.trn_info is None and trn_info is None:
            print 'No TrnInformation'
            return -1
        else:
            if not trn_info is None:
                self.trn_info = trn_info
        # checar se existe o arquivo
        file_name = '%s/%s_%s_trn_info.jbl'%(self.preproc_path,self.trn_info.date,self.name)
        if not os.path.exists(file_name):
            self.trn_info.save(file_name)
        else:
            self.trn_info.load(file_name)
        
        if self.trn_info.CVO is None:
            self.trn_info.SplitTrainSet(trgt)
            self.trn_info.save(file_name)

        train_id, test_id = self.trn_info.CVO[fold]
        
        # checar se existe o arquivo
        file_name = '%s/%s_%s_preproc_fold_%i.jbl'%(self.preproc_path,
                                                    self.trn_info.date,
                                                    self.name,fold)
        if not os.path.exists(file_name):
            print 'NeuralClassication preprocess function: creating scaler for fold %i'%(fold)
            # normalize data based in train set
            if self.trn_info.norm == 'mapstd':
                scaler = preprocessing.StandardScaler().fit(data[train_id,:])
            elif self.trn_info.norm == 'mapstd_rob':
                scaler = preprocessing.RobustScaler().fit(data[train_id,:])
            elif self.trn_info.norm == 'mapminmax':
                scaler = preprocessing.MinMaxScaler().fit(data[train_id,:])
            joblib.dump([scaler],file_name,compress=9)
        else:
            print 'NeuralClassication preprocess function: loading scaler for fold %i'%(fold)
            [scaler] = joblib.load(file_name)
        
        data_proc = scaler.transform(data)

        # other preprocessing

        return [data_proc,trgt]
            
    def train(self, data, trgt):
        return -1
