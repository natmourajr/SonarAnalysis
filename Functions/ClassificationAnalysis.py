"""
    This file contents some classification analysis functions
"""

import os
import time
import numpy as np

from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras.callbacks as callbacks
from keras.models import load_model

import matplotlib.pyplot as plt

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rc('legend',**{'fontsize':15})
plt.rc('font', weight='bold')

class TrnInformation(object):
    def __init__(self, date='', n_folds=2, n_inits=2,
                 norm='mapstd',
                 verbose=False,
                 train_verbose = False,
                 n_epochs=10,
                 learning_rate=0.01,
                 learning_decay=1e-6,
                 momentum=0.3,
                 nesterov=True,
                 patience=5,
                 batch_size=4):
        
        self.n_folds = n_folds
        self.n_inits = n_inits
        self.norm = norm
        self.verbose = verbose
        self.train_verbose = train_verbose
        
        
        # train params
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.patience = patience
        self.batch_size = batch_size
        
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
            print '\tCVO is not None'
        if verbose:
            print '\tVerbose is True'
        else:
            print '\tVerbose is False'
        if train_verbose:
            print '\tTrain Verbose is True'
        else:
            print '\tTrain Verbose is False'


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
            # checar se existe o arquivo
            file_name = '%s/%s_%s_trn_info.jbl'%(self.preproc_path,self.trn_info.date,self.name)
        
            if not os.path.exists(file_name):
                print 'No TrnInformation'
                return -1
            else:
                self.trn_info.load(file_name)
        else:
            if not trn_info is None:
                self.trn_info = trn_info
                # checar se existe o arquivo
                file_name = '%s/%s_%s_trn_info.jbl'%(self.preproc_path,
                                                     self.trn_info.date,
                                                     self.name)
                if not os.path.exists(file_name):
                    self.trn_info.save(file_name)
                        
        if self.trn_info.CVO is None:
            print 'No Cross Validation Obj'
            return -1

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
            
    def train(self, data, trgt, n_neurons=1, trn_info=None, fold=0):
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
        
                sgd = SGD(lr=self.trn_info.learning_rate,
                          decay=self.trn_info.learning_decay,
                          momentum=self.trn_info.momentum,
                          nesterov=self.trn_info.nesterov)
                model.compile(loss='mean_squared_error',
                              optimizer=sgd,
                              metrics=['accuracy'])
            
                # Train model
                earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=self.trn_info.patience,
                                                        verbose=self.trn_info.train_verbose,
                                                        mode='auto')
                                                    
                init_trn_desc = model.fit(data_preproc[train_id], trgt_preproc[train_id],
                                          nb_epoch=self.trn_info.n_epochs,
                                          batch_size=self.trn_info.batch_size,
                                          callbacks=[earlyStopping],
                                          verbose=self.trn_info.verbose,
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
            file_name = '%s/%s_%s_train_fold_%i_neurons_%i_model.h5'%(self.train_path,
                                                                      self.trn_info.date,
                                                                      self.name,fold,n_neurons)
            best_model.save(file_name)
        
            # salvar o descritor
            file_name = '%s/%s_%s_train_fold_%i_neurons_%i_trn_desc.jbl'%(self.train_path,
                                                                         self.trn_info.date,
                                                                         self.name,fold,n_neurons)
            joblib.dump([best_desc],file_name,compress=9)
        else:
            # carregar o modelo
            file_name = '%s/%s_%s_train_fold_%i_neurons_%i_model.h5'%(self.train_path,
                                                                      self.trn_info.date,
                                                                      self.name,fold,n_neurons)
            best_model = load_model(file_name)
        
            # carregar o descritor
            file_name = '%s/%s_%s_train_fold_%i_neurons_%i_trn_desc.jbl'%(self.train_path,
                                                                     self.trn_info.date,
                                                                     self.name,fold,n_neurons)
            [best_desc] = joblib.load(file_name)
        
        
        return [best_model, best_desc]
            
    def analysis(self, data, trgt, trn_info=None, n_neurons=1, fold=0):
        print 'NeuralClassication analysis function'
        #self.analysis_output_hist(data,trgt,trn_info=trn_info,n_neurons=n_neurons,fold=fold)
        #self.analysis_top_sweep(self, data, trgt, trn_info=trn_info, min_neurons=1, max_neurons=15)
        return 0
    
    def analysis_output_hist(self, data, trgt, trn_info=None, n_neurons=1, fold=0):
        print 'NeuralClassication analysis output hist function'
        # checar se a analise ja foi feita
        file_name = '%s/%s_%s_analysis_model_output_fold_%i_neurons_%i.jbl'%(self.anal_path,
                                                                            self.trn_info.date,
                                                                            self.name,fold,
                                                                            n_neurons)
        output = None
        if not os.path.exists(file_name):
            [model,trn_desc] = self.train(data,trgt,trn_info=trn_info,n_neurons=n_neurons,fold=fold)
            output = model.predict(data)
            joblib.dump([output],file_name,compress=9)
        else:
            [output] = joblib.load(file_name)
    
        fig, ax = plt.subplots(figsize=(10,10),nrows=trgt.shape[1], ncols=output.shape[1])
        
        m_colors = ['b', 'r', 'g', 'y']
        m_bins = np.linspace(-1.5, 1.5, 50)
        
        for i_target in range(trgt.shape[1]):
            for i_output in range(output.shape[1]):
                subplot_id = output.shape[1]*i_target+i_output
                # alvos max esparsos
                m_pts = output[np.argmax(trgt,axis=1)==i_target,i_output]
                n, bins, patches = ax[i_target,i_output].hist(m_pts,bins=m_bins,
                                                              fc=m_colors[i_target],
                                                              alpha=0.8, normed=1)
                if i_output == 0:
                    ax[i_target,i_output].set_ylabel('Target %i'%(i_target+1),
                                                     fontweight='bold',fontsize=15)
                if i_target == trgt.shape[1]-1:
                    ax[i_target,i_output].set_xlabel('Output %i'%(i_output+1),
                                                     fontweight='bold',fontsize=15)
                ax[i_target,i_output].grid()
        
        return fig

    def analysis_top_sweep(self, data, trgt, trn_info=None, min_neurons=1, max_neurons=2):
        print 'NeuralClassication analysis top sweep function'
        # checar se a analise ja foi feita
        file_name = '%s/%s_%s_analysis_top_sweep_min_%i_max_%i.jbl'%(self.anal_path,
                                                                     self.trn_info.date,
                                                                     self.name,
                                                                     min_neurons,
                                                                     max_neurons)
        if not os.path.exists(file_name):
            acc_vector = np.zeros([self.trn_info.n_folds,max_neurons+1])
            for ineuron in xrange(min_neurons,max_neurons+1):
                for ifold in range(self.trn_info.n_folds):
                    
                    [model,trn_desc] = self.train(data,trgt,trn_info=trn_info,n_neurons=ineuron,fold=ifold)
                    acc_vector[ifold, ineuron] = np.min(trn_desc['val_loss'])

            joblib.dump([acc_vector],file_name,compress=9)
        else:
            [acc_vector] = joblib.load(file_name)

        fig, ax = plt.subplots(figsize=(10,10),nrows=1, ncols=1)
        xtick = range(max_neurons+1)
        ax.errorbar(xtick,np.mean(acc_vector,axis=0),np.std(acc_vector,axis=0),fmt='o-',
                    color='k',alpha=0.7,linewidth=2.5)
        ax.set_ylabel('Acc',fontweight='bold',fontsize=15)
        ax.set_xlabel('Neurons',fontweight='bold',fontsize=15)
        ax.grid()
        ax.xaxis.set_ticks(xtick)
        
        return fig

    def analysis_train(self,data,trgt,trn_info=None, n_neurons=1,fold=0):
        print 'NeuralClassication analysis output hist function'
        # checar se a analise ja foi feita
        file_name = '%s/%s_%s_analysis_trn_desc_fold_%i_neurons_%i.jbl'%(self.anal_path,
                                                                         self.trn_info.date,
                                                                         self.name,fold,
                                                                         n_neurons)

        trn_desc = None
        if not os.path.exists(file_name):
            [model,trn_desc] = self.train(data,trgt,trn_info=trn_info,n_neurons=n_neurons,fold=fold)
            joblib.dump([trn_desc],file_name,compress=9)
        else:
            [trn_desc] = joblib.load(file_name)

        fig, ax = plt.subplots(figsize=(10,10),nrows=1, ncols=1)

        ax.plot(trn_desc['epochs'],trn_desc['loss'],color=[0,0,1],
                linewidth=2.5,linestyle='solid',label='Train Perf.')

        ax.plot(trn_desc['epochs'],trn_desc['val_loss'],color=[1,0,0],
                linewidth=2.5,linestyle='dashed',label='Val Perf.')

        ax.set_ylabel('MSE',fontweight='bold',fontsize=15)
        ax.set_xlabel('Epochs',fontweight='bold',fontsize=15)
        
        ax.grid()
        plt.legend()

        return fig


    def analysis_conf_mat(self,data,trgt,trn_info=None, class_labels=None, n_neurons=1,fold=0):
        print 'NeuralClassication analysis analysis conf mat function'
        file_name = '%s/%s_%s_analysis_model_output_fold_%i_neurons_%i.jbl'%(self.anal_path,
                                                                             self.trn_info.date,
                                                                             self.name,fold,
                                                                             n_neurons)
        output = None
        if not os.path.exists(file_name):
            [model,trn_desc] = self.train(data,trgt,trn_info=trn_info,n_neurons=n_neurons,fold=fold)
            output = model.predict(data)
            joblib.dump([output],file_name,compress=9)
        else:
            [output] = joblib.load(file_name)

        fig, ax = plt.subplots(figsize=(10,10),nrows=1, ncols=1)


        train_id, test_id = self.trn_info.CVO[fold]

        num_output = np.argmax(output,axis=1)
        num_tgrt = np.argmax(trgt,axis=1)

        cm = confusion_matrix(num_tgrt[test_id], num_output[test_id])
        cm_normalized = 100.*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        im =ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Greys,clim=(0.0, 100.0))

        width, height = cm_normalized.shape

        for x in xrange(width):
            for y in xrange(height):
                if cm_normalized[x][y] < 50.:
                    ax.annotate('%1.3f%%'%(cm_normalized[x][y]), xy=(y, x),
                                horizontalalignment='center',
                                verticalalignment='center')
                else:
                    ax.annotate('%1.3f%%'%(cm_normalized[x][y]), xy=(y, x),
                                horizontalalignment='center',
                                verticalalignment='center',color='white')
        ax.set_title('Confusion Matrix',fontweight='bold',fontsize=15)
        fig.colorbar(im)
        if not class_labels is None:
            tick_marks = np.arange(len(class_labels))
            ax.xaxis.set_ticks(tick_marks)
            ax.xaxis.set_ticklabels(class_labels)

            ax.yaxis.set_ticks(tick_marks)
            ax.yaxis.set_ticklabels(class_labels)

        return fig



