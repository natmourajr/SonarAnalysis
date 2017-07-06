"""
    This file contents some classification analysis functions
"""

import os
import time
import numpy as np

from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn import preprocessing


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras.callbacks as callbacks
from keras.models import load_model


class TrnInformation(object):
    def __init__(self, date='', n_folds=2, n_inits=2, n_neurons=4,
                 norm='mapstd',verbose=False,
                 train_verbose = False):
        self.n_folds = n_folds
        self.n_inits = n_inits
        self.n_neurons = n_neurons
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
        print '\tNumber of Neurons: %i'%(self.n_neurons)
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
        
        if fold > trn_info.n_folds or fold < -1:
            print 'Invalid Fold...'
            return None
        
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
            self.preproc_done = True
        else:
            #print 'NeuralClassication preprocess function: loading scaler for fold %i'%(fold)
            [scaler] = joblib.load(file_name)
        
        data_proc = scaler.transform(data)

        # others preprocessing process

        return [data_proc,trgt]
            
    def train(self, data, trgt, n_neurons=4, trn_info=None, fold=0):
        print 'NeuralClassication train function'
        
        if fold > trn_info.n_folds or fold < -1:
            print 'Invalid Fold...'
            return None
        
        [data_preproc, trgt_preproc] = self.preprocess(data,trgt,
                                                       trn_info=trn_info,fold=fold)
                                                       
        # checar se o arquivo existe
        file_name = '%s/%s_%s_train_fold_%i_neurons_%i_model.h5'%(self.preproc_path,
                                                                  self.trn_info.date,
                                                                  self.name,fold,n_neurons)
        if not os.path.exists(file_name):
        
            best_init = 0
            best_loss = 999
            best_model = None
            best_desc = {}
        
            train_id, test_id = self.trn_info.CVO[fold]
            
            for i_init in range(self.trn_info.n_inits):
                print 'Init: %i of %i'%(i_init+1,self.trn_info.n_inits)
                
                model = Sequential()
                model.add(Dense(n_neurons, input_dim=data.shape[1], init="uniform"))
                model.add(Activation('tanh'))
                model.add(Dense(trgt.shape[1], init="uniform"))
                model.add(Activation('tanh'))
        
                sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                model.compile(loss='mean_squared_error',
                              optimizer=sgd,
                              metrics=['accuracy'])
            
                # Train model
                earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=5,
                                                        verbose=0,
                                                        mode='auto')
                                                    
                init_trn_desc = model.fit(data_preproc[train_id], trgt_preproc[train_id],
                                          nb_epoch=3,
                                          batch_size=256,
                                          callbacks=[earlyStopping],
                                          verbose=1,
                                          validation_data=(data_preproc[test_id],
                                                           trgt_preproc[test_id]),
                                          shuffle=True)
        
                if np.min(init_trn_desc.history['val_loss']) < best_loss:
                    best_init = i_init
                    best_loss = np.min(init_trn_desc.history['val_loss'])
                    best_model = model
                    best_desc['epochs'] = init_trn_desc.epoch
                    best_desc['acc'] = init_trn_desc.history['acc']
                    best_desc['loss'] = init_trn_desc.history['loss']
                    best_desc['val_loss'] = init_trn_desc.history['val_loss']
                    best_desc['val_acc'] = init_trn_desc.history['val_acc']
        
            # salvar o modelo
            file_name = '%s/%s_%s_train_fold_%i_neurons_%i_model.h5'%(self.preproc_path,
                                                                      self.trn_info.date,
                                                                      self.name,fold,n_neurons)
            best_model.save(file_name)
        
            # salvar o descritor
            file_name = '%s/%s_%s_train_fold_%i_neurons_%i_trn_desc.h5'%(self.preproc_path,
                                                                         self.trn_info.date,
                                                                         self.name,fold,n_neurons)
            joblib.dump([best_desc],file_name,compress=9)
        else:
            # salvar o modelo
            file_name = '%s/%s_%s_train_fold_%i_neurons_%i_model.h5'%(self.preproc_path,
                                                                      self.trn_info.date,
                                                                      self.name,fold,n_neurons)
            best_model = load_model(file_name)
        
        
        return 0
