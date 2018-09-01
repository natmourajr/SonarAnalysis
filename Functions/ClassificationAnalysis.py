"""
    This file contents some classification analysis functions
"""
import gc
import os
import time
from collections import OrderedDict
from itertools import cycle

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
import keras.callbacks as callbacks
from keras.models import load_model

from keras import backend as backend

import matplotlib.pyplot as plt

from Functions.ConvolutionalNeuralNetworks import KerasModel
from Functions.FunctionsDataVisualization import plotConfusionMatrix, plotLOFARgram
from Functions.NpUtils.DataTransformation import SonarRunsInfo, lofar2image
from Functions.NpUtils.Scores import recall_score, spIndex
from Functions.SystemIO import exists, mkdir, save, load

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rc('legend', **{'fontsize': 15})
plt.rc('font', weight='bold')


class TrnInformation(object):
    def __init__(self, date='', n_folds=2, n_inits=2,
                 norm='mapstd',
                 verbose=False,
                 train_verbose=False,
                 n_epochs=10,
                 learning_rate=0.001,
                 learning_decay=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-08,
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
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.patience = patience
        self.batch_size = batch_size

        self.CVO = None
        if date == '':
            self.date = time.strftime("%Y_%m_%d_%H_%M_%S")
        else:
            self.date = date

    def Print(self):
        print 'Class TrnInformation'
        print '\tDate %s' % (self.date)
        print '\tNumber of Folds %i' % (self.n_folds)
        print '\tNumber of Initializations: %i' % (self.n_inits)
        print '\tNormalization: %s' % (self.norm)
        if self.CVO is None:
            print '\tCVO is None'
        else:
            print '\tCVO is not None'
        if self.verbose:
            print '\tVerbose is True'
        else:
            print '\tVerbose is False'
        if self.train_verbose:
            print '\tTrain Verbose is True'
        else:
            print '\tTrain Verbose is False'

    #	if self.train_done:
    #	    print '\tTrain is done'
    #	else:
    #	    print '\tTrain is not done'

    def SplitTrainSet(self, trgt):
        # divide data in train and test for novelty detection
        CVO = cross_validation.StratifiedKFold(trgt, self.n_folds)
        self.CVO = list(CVO)

    def save(self, path=''):
        print 'Save TrnInformation'
        if path == '':
            print 'No valid path...'
            return -1
        joblib.dump([self.date, self.n_folds, self.n_inits, self.norm, self.CVO], path, compress=9)
        return 0

    def load(self, path):
        print 'Load TrnInformation'
        if not os.path.exists(path):
            print 'No valid path...'
            return -1
        [self.date, self.n_folds, self.n_inits, self.norm, self.CVO] = joblib.load(path)


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

    def Print(self):
        print 'Class %s' % (self.name)
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
            self.preprocess(data, trgt)

        # train process
        self.training_done = True

    def analysis(self, data, trgt):
        print 'ClassificationAnalysis analysis function'
        if not self.training_done:
            self.train(data, trgt)

        # analysis process
        self.analysis_done = True


class NeuralClassification(ClassificationBaseClass):
    def preprocess(self, data, trgt, trn_info=None, fold=0):
        print 'NeuralClassication preprocess function'

        if fold > trn_info.n_folds or fold < -1:
            print 'Invalid Fold...'
            return None

        if self.trn_info is None and trn_info is None:
            # Check if the file exist
            file_name = '%s/%s_%s_trn_info.jbl' % (self.preproc_path, self.trn_info.date, self.name)

            if not os.path.exists(file_name):
                print 'No TrnInformation'
                return -1
            else:
                self.trn_info.load(file_name)
        else:
            if not trn_info is None:
                self.trn_info = trn_info
                # Check if the file exist
                file_name = '%s/%s_%s_trn_info.jbl' % (self.preproc_path,
                                                       self.trn_info.date,
                                                       self.name)
                if not os.path.exists(file_name):
                    self.trn_info.save(file_name)

        if self.trn_info.CVO is None:
            print 'No Cross Validation Obj'
            return -1

        train_id, test_id = self.trn_info.CVO[fold]

        # Check if the file exist
        file_name = '%s/%s_%s_preproc_fold_%i.jbl' % (self.preproc_path,
                                                      self.trn_info.date,
                                                      self.name, fold)
        if not os.path.exists(file_name):
            print 'NeuralClassication preprocess function: creating scaler for fold %i' % (fold)
            # normalize data based in train set
            if self.trn_info.norm == 'mapstd':
                scaler = preprocessing.StandardScaler().fit(data[train_id, :])
            elif self.trn_info.norm == 'mapstd_rob':
                scaler = preprocessing.RobustScaler().fit(data[train_id, :])
            elif self.trn_info.norm == 'mapminmax':
                scaler = preprocessing.MinMaxScaler().fit(data[train_id, :])
            joblib.dump([scaler], file_name, compress=9)
            self.preproc_done = True
        else:
            print 'NeuralClassication preprocess function: loading scaler for fold %i' % (fold)
            [scaler] = joblib.load(file_name)

        data_proc = scaler.transform(data)

        # others preprocessing process
        return [data_proc, trgt]

    def train(self, data, trgt, n_neurons=1, trn_info=None, fold=0):
        print 'NeuralClassication train function'

        if fold > trn_info.n_folds or fold < -1:
            print 'Invalid Fold...'
            return None

        [data_preproc, trgt_preproc] = self.preprocess(data, trgt,
                                                       trn_info=trn_info, fold=fold)
        # Check if the file exists
        file_name = '%s/%s_%s_train_fold_%i_neurons_%i_model.h5' % (self.preproc_path,
                                                                    self.trn_info.date,
                                                                    self.name, fold, n_neurons)
        if not os.path.exists(file_name):
            best_init = 0
            best_loss = 999
            best_model = None
            best_desc = {}

            train_id, test_id = self.trn_info.CVO[fold]

            for i_init in range(self.trn_info.n_inits):
                print 'Init: %i of %i' % (i_init + 1, self.trn_info.n_inits)

                model = Sequential()
                model.add(Dense(n_neurons, input_dim=data.shape[1], init="uniform"))
                model.add(Activation('softplus'))
                model.add(Dense(trgt.shape[1], init="uniform"))
                model.add(Activation('softmax'))

                adam = Adam(lr=self.trn_info.learning_rate,
                            # decay  = self.trn_info.learning_decay,
                            beta_1=self.trn_info.beta_1,
                            beta_2=self.trn_info.beta_2,
                            epsilon=self.trn_info.epsilon)

                model.compile(loss='mean_squared_error',
                              optimizer='Adam',
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

            # Save the model
            file_name = '%s/%s_%s_train_fold_%i_neurons_%i_model.h5' % (self.train_path,
                                                                        self.trn_info.date,
                                                                        self.name, fold, n_neurons)
            best_model.save(file_name)

            # Save the descriptor
            file_name = '%s/%s_%s_train_fold_%i_neurons_%i_trn_desc.jbl' % (self.train_path,
                                                                            self.trn_info.date,
                                                                            self.name, fold, n_neurons)
            joblib.dump([best_desc], file_name, compress=9)
        else:
            # Load the model
            file_name = '%s/%s_%s_train_fold_%i_neurons_%i_model.h5' % (self.train_path,
                                                                        self.trn_info.date,
                                                                        self.name, fold, n_neurons)
            best_model = load_model(file_name)

            # Load the descriptor
            file_name = '%s/%s_%s_train_fold_%i_neurons_%i_trn_desc.jbl' % (self.train_path,
                                                                            self.trn_info.date,
                                                                            self.name, fold, n_neurons)
            [best_desc] = joblib.load(file_name)

        return [best_model, best_desc]

    def analysis(self, data, trgt, trn_info=None, n_neurons=1, fold=0):
        print 'NeuralClassication analysis function'
        # self.analysis_output_hist(data,trgt,trn_info=trn_info,n_neurons=n_neurons,fold=fold)
        # self.analysis_top_sweep(self, data, trgt, trn_info=trn_info, min_neurons=1, max_neurons=15)
        return 0

    def analysis_output_hist(self, data, trgt, trn_info=None, n_neurons=1, fold=0):
        print 'NeuralClassication analysis output hist function'
        # Check if the analysis has already been done
        file_name = '%s/%s_%s_analysis_model_output_fold_%i_neurons_%i.jbl' % (self.anal_path,
                                                                               self.trn_info.date,
                                                                               self.name, fold,
                                                                               n_neurons)
        output = None
        if not os.path.exists(file_name):
            [model, trn_desc] = self.train(data, trgt, trn_info=trn_info, n_neurons=n_neurons, fold=fold)
            output = model.predict(data)
            joblib.dump([output], file_name, compress=9)
        else:
            [output] = joblib.load(file_name)

        fig, ax = plt.subplots(figsize=(10, 10), nrows=trgt.shape[1], ncols=output.shape[1])

        m_colors = ['b', 'r', 'g', 'y']
        m_bins = np.linspace(-0.5, 1.5, 50)
        for i_target in range(trgt.shape[1]):
            for i_output in range(output.shape[1]):
                subplot_id = output.shape[1] * i_target + i_output
                # alvos max esparsos
                m_pts = output[np.argmax(trgt, axis=1) == i_target, i_output]
                print m_pts
                n, bins, patches = ax[i_target, i_output].hist(m_pts, bins=m_bins,
                                                               fc=m_colors[i_target],
                                                               alpha=0.8, normed=1)
                if i_output == 0:
                    ax[i_target, i_output].set_ylabel('Target %i' % (i_target + 1),
                                                      fontweight='bold', fontsize=15)
                if i_target == trgt.shape[1] - 1:
                    ax[i_target, i_output].set_xlabel('Output %i' % (i_output + 1),
                                                      fontweight='bold', fontsize=15)
                ax[i_target, i_output].grid()

        return None

    def analysis_top_sweep(self, data, trgt, trn_info=None, min_neurons=1, max_neurons=2):
        print 'NeuralClassication analysis top sweep function'
        # Check if the analysis has already been done
        file_name = '%s/%s_%s_analysis_top_sweep_min_%i_max_%i.jbl' % (self.anal_path,
                                                                       self.trn_info.date,
                                                                       self.name,
                                                                       min_neurons,
                                                                       max_neurons)
        if not os.path.exists(file_name):
            acc_vector = np.zeros([self.trn_info.n_folds, max_neurons + 1])
            for ineuron in xrange(min_neurons, max_neurons + 1):
                for ifold in range(self.trn_info.n_folds):
                    [model, trn_desc] = self.train(data, trgt, trn_info=trn_info, n_neurons=ineuron, fold=ifold)
                    acc_vector[ifold, ineuron] = np.min(trn_desc['val_acc'])  # estava val_loss

            joblib.dump([acc_vector], file_name, compress=9)
        else:
            [acc_vector] = joblib.load(file_name)

        fig, ax = plt.subplots(figsize=(6, 6), nrows=1, ncols=1)
        xtick = range(max_neurons + 1)
        print acc_vector
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.errorbar(xtick, np.mean(acc_vector, axis=0), np.std(acc_vector, axis=0), fmt='o-',
                    color='k', alpha=0.7, linewidth=2.5)
        ax.set_ylabel('Acc', fontweight='bold', fontsize=15)
        ax.set_xlabel('Neurons', fontweight='bold', fontsize=15)

        ax.grid()
        ax.xaxis.set_ticks(xtick)

        return None

    def analysis_train_plot(self, data, trgt, trn_info=None, n_neurons=1, fold=0):
        # print 'NeuralClassication analysis train plot function'
        # Check if the analysis has already been done
        file_name = '%s/%s_%s_analysis_trn_desc_fold_%i_neurons_%i.jbl' % (self.anal_path,
                                                                           self.trn_info.date,
                                                                           self.name, fold,
                                                                           n_neurons)

        trn_desc = None
        if not os.path.exists(file_name):
            [model, trn_desc] = self.train(data, trgt, trn_info=trn_info, n_neurons=n_neurons, fold=fold)
            joblib.dump([trn_desc], file_name, compress=9)
        else:
            [trn_desc] = joblib.load(file_name)

        # print "Results for Fold %i:"%fold
        fig, ax = plt.subplots(figsize=(6, 6), nrows=1, ncols=1)

        ax.plot(trn_desc['epochs'], trn_desc['loss'], color=[0, 0, 1],
                linewidth=2.5, linestyle='solid', label='Train Perf.')

        ax.plot(trn_desc['epochs'], trn_desc['val_loss'], color=[1, 0, 0],
                linewidth=2.5, linestyle='dashed', label='Val Perf.')

        ax.set_ylabel('MSE', fontweight='bold', fontsize=15)
        ax.set_xlabel('Epochs', fontweight='bold', fontsize=15)

        ax.grid()
        plt.legend()

        return None

    def analysis_conf_mat(self, data, trgt, trn_info=None, class_labels=None, n_neurons=1, fold=0):
        # print 'NeuralClassication analysis analysis conf mat function'
        file_name = '%s/%s_%s_analysis_model_output_fold_%i_neurons_%i.jbl' % (self.anal_path,
                                                                               self.trn_info.date,
                                                                               self.name, fold,
                                                                               n_neurons)
        output = None
        # If the file doesn't exists, train new model and save it
        # Check if the file exists
        if not os.path.exists(file_name):
            # If the file doesn't exists, train new model and save it
            [model, trn_desc] = self.train(data, trgt, trn_info=trn_info, n_neurons=n_neurons, fold=fold)
            output = model.predict(data)
            joblib.dump([output], file_name, compress=9)
        else:
            [output] = joblib.load(file_name)

        # print "Results for Fold %i:"%fold

        fig, ax = plt.subplots(figsize=(6, 6), nrows=1, ncols=1)

        train_id, test_id = self.trn_info.CVO[fold]
        num_output = np.argmax(output, axis=1)
        num_tgrt = np.argmax(trgt, axis=1)

        cm = confusion_matrix(num_tgrt[test_id], num_output[test_id])
        cm_normalized = 100. * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Greys, clim=(0.0, 100.0))

        width, height = cm_normalized.shape

        for x in xrange(width):
            for y in xrange(height):
                if cm_normalized[x][y] < 50.:
                    ax.annotate('%1.3f%%' % (cm_normalized[x][y]), xy=(y, x),
                                horizontalalignment='center',
                                verticalalignment='center')
                else:
                    ax.annotate('%1.3f%%' % (cm_normalized[x][y]), xy=(y, x),
                                horizontalalignment='center',
                                verticalalignment='center', color='white')
        ax.set_title('Confusion Matrix', fontweight='bold', fontsize=15)
        fig.colorbar(im)
        if not class_labels is None:
            tick_marks = np.arange(len(class_labels))
            ax.xaxis.set_ticks(tick_marks)
            ax.xaxis.set_ticklabels(class_labels)

            ax.yaxis.set_ticks(tick_marks)
            ax.yaxis.set_ticklabels(class_labels)
        ax.set_ylabel('True Label', fontweight='bold', fontsize=15)
        ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=15)
        return None

    def accuracy(self, data, trgt, trn_info=None, threshold=0.5, n_neurons=1, fold=0):
        file_name = '%s/%s_%s_analysis_model_output_fold_%i_neurons_%i.jbl' % (self.anal_path,
                                                                               self.trn_info.date,
                                                                               self.name, fold,
                                                                               n_neurons)
        output = None
        # If the file doesn't exists, train new model and save it
        # Check if the file exists
        if not os.path.exists(file_name):
            # If the file doesn't exists, train new model and save it
            [model, trn_desc] = self.train(data, trgt, trn_info=trn_info, n_neurons=n_neurons, fold=fold)
            output = model.predict(data)
            joblib.dump([output], file_name, compress=9)
        else:
            [output] = joblib.load(file_name)

    def analysis_train_plot(self, data, trgt, trn_info=None, n_neurons=1, fold=0):
        print 'NeuralClassication analysis train plot function'
        # checar se a analise ja foi feita
        file_name = '%s/%s_%s_analysis_trn_desc_fold_%i_neurons_%i.jbl' % (self.anal_path,
                                                                           self.trn_info.date,
                                                                           self.name, fold,
                                                                           n_neurons)

        trn_desc = None
        if not os.path.exists(file_name):
            [model, trn_desc] = self.train(data, trgt, trn_info=trn_info, n_neurons=n_neurons, fold=fold)
            joblib.dump([trn_desc], file_name, compress=9)
        else:
            [trn_desc] = joblib.load(file_name)

        fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1)

        ax.plot(trn_desc['epochs'], trn_desc['loss'], color=[0, 0, 1],
                linewidth=2.5, linestyle='solid', label='Train Perf.')

        ax.plot(trn_desc['epochs'], trn_desc['val_loss'], color=[1, 0, 0],
                linewidth=2.5, linestyle='dashed', label='Val Perf.')

        ax.set_ylabel('MSE', fontweight='bold', fontsize=15)
        ax.set_xlabel('Epochs', fontweight='bold', fontsize=15)

        ax.grid()
        plt.legend()

        return fig

    def analysis_conf_mat(self, data, trgt, trn_info=None, class_labels=None, n_neurons=1, fold=0):
        print 'NeuralClassication analysis analysis conf mat function'
        file_name = '%s/%s_%s_analysis_model_output_fold_%i_neurons_%i.jbl' % (self.anal_path,
                                                                               self.trn_info.date,
                                                                               self.name, fold,
                                                                               n_neurons)
        output = None
        if not os.path.exists(file_name):
            [model, trn_desc] = self.train(data, trgt, trn_info=trn_info, n_neurons=n_neurons, fold=fold)
            output = model.predict(data)
            joblib.dump([output], file_name, compress=9)
        else:
            [output] = joblib.load(file_name)

        fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1)

        train_id, test_id = self.trn_info.CVO[fold]

        num_output = np.argmax(output, axis=1)
        num_tgrt = np.argmax(trgt, axis=1)

        cm = confusion_matrix(num_tgrt[test_id], num_output[test_id])
        cm_normalized = 100. * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Greys, clim=(0.0, 100.0))

        width, height = cm_normalized.shape

        for x in xrange(width):
            for y in xrange(height):
                if cm_normalized[x][y] < 50.:
                    ax.annotate('%1.3f%%' % (cm_normalized[x][y]), xy=(y, x),
                                horizontalalignment='center',
                                verticalalignment='center')
                else:
                    ax.annotate('%1.3f%%' % (cm_normalized[x][y]), xy=(y, x),
                                horizontalalignment='center',
                                verticalalignment='center', color='white')
        ax.set_title('Confusion Matrix', fontweight='bold', fontsize=15)
        fig.colorbar(im)
        if not class_labels is None:
            tick_marks = np.arange(len(class_labels))
            ax.xaxis.set_ticks(tick_marks)
            ax.xaxis.set_ticklabels(class_labels)

            ax.yaxis.set_ticks(tick_marks)
            ax.yaxis.set_ticklabels(class_labels)

        return fig

        # return K.mean(K.equal(trgt, K.lesser(output, threshold)))


class CnnAnalysisFunction:
    def __init__(self, ncv_obj, trnParamsMapping, package_name, analysis_name, class_labels):
        self.modelsData = {model_name: ModelDataCollection(ncv_obj, trnParams, package_name,
                                                           analysis_name + '/%s' % model_name,
                                                           class_labels)
                              for model_name, trnParams in trnParamsMapping.items()}
        for model_name,cnn_an in self.modelsData.items():
            print model_name
            print cnn_an.params.input_shape

        for modelData in self.modelsData.values():
            modelData.fetchPredictions()
            modelData.fecthHistory()

        self.an_path = package_name + '/' + analysis_name
        self.resultspath = package_name
        self.class_labels = class_labels
        self.scores = None
        self.getScores()

    def stdAnalysis(self, data, trgt):
        for modelData in self.modelsData.values():
            modelData.fetchPredictions()
            modelData.fecthHistory()
            modelData.plotConfusionMatrices()
            modelData.plotTraining()
            modelData.getScores()
            modelData.plotDensities()
            modelData.plotRuns(data, trgt, ["Conv2D"], overwrite=False)

    def getScores(self):
        scores = {model_name:cnn_an.getScores() for model_name, cnn_an in self.modelsData.items()}

        for model_name in scores.keys():
            scores[model_name]['Model'] = model_name

        self.scores = pd.concat(scores, axis=0)

    def plotScores(self):
        markers = ['^', 'o', '+', 's', 'p', 'o', '8', 'D', 'x']
        linestyles = ['-', '-', ':', '-.']
        colors = ['k', 'b', 'g', 'y', 'r', 'm', 'y', 'w']

        def cndCycler(cycler, std_marker, condition, data):
            return [std_marker if condition(var) else cycler.next() for var in data]

        for cv_name, cv_results in self.scores.groupby(level='CV'):
            scorespath = self.an_path + '/%s.pdf' % cv_name

            sns.set_style("whitegrid")

            fig, ax = plt.subplots(figsize=(15, 8), nrows=1, ncols=1)

            plt.rcParams['xtick.labelsize'] = 15
            plt.rcParams['ytick.labelsize'] = 15
            plt.rcParams['legend.numpoints'] = 1
            plt.rc('legend', **{'fontsize': 15})
            plt.rc('font', weight='bold')

            cv_results = cv_results.melt(id_vars=['Model'])
            sns.pointplot(y='value', x='Model', hue='variable',
                          data=cv_results,
                          markers=cndCycler(cycle(markers[:-1]),
                                            markers[-1],
                                            lambda x: x in ['Eff %s' % cls_name for cls_name in self.class_labels.values()],
                                            cv_results['variable'].unique()),
                          linestyles=cndCycler(cycle(linestyles[:-1]), linestyles[-1],
                                               lambda x: x in ['Eff %s' % cls_name for cls_name in
                                                               self.class_labels.values()],
                                               cv_results['variable'].unique()),
                          palette=cndCycler(cycle(colors[1:]), colors[0],
                                            lambda x: not x in ['Eff %s' % cls_name for cls_name in
                                                                self.class_labels.values()],
                                            cv_results['variable'].unique()),
                          dodge=.5,
                          scale=1.7,
                          errwidth=2.2, capsize=.1, ax=ax)
            leg_handles = ax.get_legend_handles_labels()[0]
            ax.legend(handles=leg_handles,
                      ncol=6, mode="expand", borderaxespad=0., loc=3)
            ax.set_xlabel("Window Quantity", fontsize=20, weight='bold')
            # ax.set_ylabel('Figures of Merit', fontsize=20, weight='bold')
            ax.set_title('Classification Efficiency for Different Window Quantities', fontsize=25, weight='bold')
            plt.yticks(weight='bold', fontsize=16)
            plt.xticks(ha='right', weight='bold', fontsize=16)
            ax2 = ax.twinx()
            ax.set_ylabel('Classification Efficiency', fontsize=20, weight='bold')
            ax2.set_ylabel('SP index', fontsize=20, weight='bold')
            ax2.set_ylim([.0, 1.0001])
            ax.set_ylim([.0, 1.0001])
            plt.yticks(weight='bold', fontsize=16)
            fig.savefig(scorespath, bbox_inches='tight')
            plt.close(fig)

    def fetchPredictions(self):
        raise NotImplementedError

    def fetchHistory(self):
        raise NotImplementedError

    def plotConfusionMatrices(self):
        raise NotImplementedError

    def plotConfusionGrid(self):
        raise NotImplementedError

    def plotTraining(self):
        raise NotImplementedError

    def plotDensitise(self):
        raise NotImplementedError

    def plotRuns(self):
        raise NotImplementedError

class ModelDataCollection:
    def __init__(self, ncv_obj, trnParams, package_name, analysis_name, class_labels):
        self.n_cv = ncv_obj
        self.params = trnParams
        # TODO add already trained folds check
        self.an_path = package_name + '/' + analysis_name
        self.resultspath = package_name
        self.modelpath = package_name + '/' + trnParams.getParamPath()
        self.trained_cvs = self.n_cv.cv.items()
        self.class_labels = class_labels
        self.history = dict()
        self.predictions = dict()

        if not exists(self.an_path):
            mkdir(self.an_path)

    def fecthHistory(self, overwrite=False):
        collections_filepath = self.an_path + '/history_collection.jbl'
        if exists(collections_filepath) and not overwrite:
            print("Collection found on analysis folder. Loading existing predictions. "
                  "To overwrite existing configuration, set overwrite to True")
            self.history = load(collections_filepath)
            return
        for cv_name, cv in self.n_cv.cv.items():
            fold_path = self.modelpath + '/%s/history.csv' % cv_name
            self.history[cv_name] = pd.read_csv(fold_path,
                                                index_col=[0, 1, 2])
            save(self.history, collections_filepath)

    def fetchPredictions(self, overwrite=False):
        collections_filepath = self.an_path + '/predictions_collection.jbl'
        if exists(collections_filepath) and not overwrite:
            print("Collection found on analysis folder.Loading existing predictions. "
                  "To overwrite existing configuration, set overwrite to True")
            self.predictions = load(collections_filepath)
            return
        for cv_name, cv in self.n_cv.cv.items():
            fold_path = self.modelpath + '/%s/predictions.csv' % cv_name

            self.predictions[cv_name] = pd.read_csv(fold_path,
                                                    index_col=[0, 2])
            self.predictions[cv_name] = self.predictions[cv_name].drop(columns='Unnamed: 1')
            self.predictions[cv_name].index.rename(['Fold', 'Sample'], level=[0,1], inplace=True)
        save(self.predictions, collections_filepath)

    def plotConfusionMatrices(self, figsize=(10, 12), overwrite=False):
        class_labels = self.class_labels
        for cv_name, cv in self.n_cv.cv.items():
            saving_path = self.an_path + '/%s/cm/' % cv_name
            cv_prediction = self.predictions[cv_name]
            if not exists(saving_path):
                mkdir(saving_path)
            else:
                if not overwrite:
                    print "CM already plotted. For overwriting current plots, " \
                          "set overwrite to True. Exiting."
                    return

            for i_fold, fold_prediction in cv_prediction.groupby(level=[0]):
                filepath = saving_path + '%s.pdf' % i_fold
                cat_prediction = fold_prediction.drop(columns='Label').values.argmax(axis=1)

                fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)
                plotConfusionMatrix(cat_prediction, fold_prediction['Label'], class_labels, ax)

                fig.savefig(filepath, bbox_inches='tight')
                plt.close(fig)

    def plotTraining(self, scores=None, scores_labels=None, figsize=(15, 8), overwrite=False):
        if scores is None:
            scores = ['spIndex', 'val_spIndex', 'loss', 'val_loss']
        else:
            scores.extend(['val_' + score for score in scores])
            print scores
        if scores_labels is None:
            scores_labels = {'spIndex': 'Train SP Index',
                             'val_spIndex': 'Val SP Index',
                             'loss': 'Train Loss',
                             'val_loss': 'Val Loss'}
        else:
            for score in scores:
                scores_labels['val_' + score] = 'Val ' + scores_labels[score]

        for cv_name, cv in self.n_cv.cv.items():
            losspath = self.an_path + '/%s/loss/' % cv_name
            sns.set_style("whitegrid")

            if not exists(losspath):
                mkdir(losspath)
            else:
                if not overwrite:
                    print "Loss already plotted. For overwriting current plots, " \
                          "set overwrite to True. Exiting."
                    return
            model_loss = self.history[cv_name]
            model_loss = model_loss.loc[:, scores]
            model_loss = model_loss.rename(mapper=scores_labels, axis='columns')

            for i_fold, fold_loss in model_loss.groupby(level=[0]):
                fig, fom_ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)

                plt.rcParams['xtick.labelsize'] = 15
                plt.rcParams['ytick.labelsize'] = 15
                plt.rcParams['legend.numpoints'] = 1
                plt.rc('legend', **{'fontsize': 15})
                plt.rc('font', weight='bold')

                fold_loss.loc[:, 'epochs'] = range(0, fold_loss.shape[0])

                df_loss = fold_loss.loc[:, [scores_labels['loss'], scores_labels['val_loss'], 'epochs']]
                df_fom = fold_loss.drop(columns=[scores_labels['loss'], scores_labels['val_loss']])

                loss_ax = fom_ax.twinx()

                # colors = cycle(['k', 'b', 'g', 'y', 'r', 'm', 'y', 'w'])
                colors = cycle(['black', 'grey'])

                # Plot scores with different markers and same color
                for metric in df_loss.drop(columns=['epochs']):
                    loss_ax.plot(df_loss['epochs'], df_loss[metric],
                                 linestyle='-.',
                                 linewidth=4,
                                 markersize=7,
                                 marker='o',
                                 color=colors.next(),
                                 markeredgewidth=0.0
                                 )

                colors = cycle(['blue', 'orangered', 'green', 'orchid'])
                # Plot loss with different colors
                for metric in df_fom.drop(columns=['epochs']):
                    fom_ax.plot(df_fom['epochs'], df_fom[metric],
                                marker='o',
                                linewidth=3,
                                markersize=7,
                                color=colors.next(),
                                markeredgewidth=0.0
                                )

                # Sum labels from two different axes
                lines, labels = fom_ax.get_legend_handles_labels()
                lines2, labels2 = loss_ax.get_legend_handles_labels()
                fom_ax.legend(lines + lines2, labels + labels2, loc=3, ncol=df_fom.shape[1] + df_loss.shape[1])

                fom_ax.set_xlabel('Number of epochs', fontsize=20, weight='bold')
                fom_ax.set_ylabel('Figures of Merit', fontsize=20, weight='bold')
                fom_ax.set_ylim([0.0, 1.001])

                loss_ax.set_ylabel('Loss', fontsize=20, weight='bold')

                fig.savefig(losspath + '/%s.pdf' % i_fold)
                plt.close(fig)

    def getScores(self):
        class_labels = self.class_labels
        def getScoresDict(fold_predictions, class_labels):
            cat_predictions = fold_predictions.drop(columns='Label').values.argmax(axis=1)
            recall_values = recall_score(fold_predictions['Label'], cat_predictions)
            scores_dict =  {'Eff %s' % class_labels[cls_i]: recall_value
                            for cls_i, recall_value in enumerate(recall_values)}
            scores_dict['SP Index'] = spIndex(recall_values, len(class_labels.keys()))

            return scores_dict

        def getFoldScores(cv_name, cv):
            model_predictions = self.predictions[cv_name]
            model_scores = OrderedDict(((cv_name, fold_name), getScoresDict(fold_predictions, class_labels))
                                       for fold_name, fold_predictions in model_predictions.groupby(level=[0]))
            index = pd.MultiIndex.from_tuples(model_scores.keys(), names=['CV', 'Fold'])
            model_scores = pd.DataFrame(model_scores.values(), index=index)
            return model_scores

        model_scores = pd.concat([getFoldScores(cv_name, cv) for cv_name, cv in self.trained_cvs], axis=0)
        return model_scores

    def plotDensities(self, bins=50, xtick_rotation=45, hspace=0.35, wspace=0.25, overwrite=False):
        class_labels = self.class_labels
        for cv_name, cv in self.n_cv.cv.items():
            densitypath = self.an_path + '/%s/densities/' % cv_name
            sns.set_style("whitegrid")

            if not exists(densitypath):
                mkdir(densitypath)
            else:
                if not overwrite:
                    print "Densities already plotted. For overwriting current plots, " \
                          "set overwrite to True. Exiting."
                    return

            plt.style.use('seaborn-whitegrid')
            predictions = self.predictions[cv_name]
            for fold_name, fold_prediction in predictions.groupby(level='Fold'):
                fig = plt.figure(figsize=(15, 8))
                grid = plt.GridSpec(4, 4, hspace=hspace,wspace=wspace)
                colors = ['b', 'r', 'g', 'y']
                axes = np.array([[fig.add_subplot(grid[i, j]) for i in range(0,4)] for j in range(0,4)])
                for correct_label, cls_predictions in fold_prediction.groupby(by='Label'):
                    for pred_i, pred_label in enumerate(cls_predictions.drop(columns=['Label'])):
                        i, j = int(correct_label), pred_i
                        network_output = cls_predictions[pred_label].values  # Output for the predicted class
                        weights = np.zeros_like(network_output) + 1. / network_output.shape[0]  # Normalize bins
                        axes[i, j].hist(cls_predictions[pred_label].values,
                                        bins=bins,
                                        color=colors[i],
                                        weights=weights,
                                        histtype='stepfilled',
                                        alpha = .6)
                        axes[i, j].set_yscale('log')
                        axes[i, j].set_yticks(np.logspace(-3, 0, 4))
                        axes[i,j].tick_params(axis='both', which='both', labelsize=10)
                        for tick in axes[i,j].get_xticklabels():
                            tick.set_rotation(xtick_rotation)

                    axes[i,0].set_title(class_labels[i], fontsize = 15)
                    axes[0,i].set_ylabel(class_labels[i], fontsize = 15)
                fig.savefig(densitypath + '/%s.pdf' % fold_name, bbox_inches='tight')
                plt.close(fig)

    def plotRuns(self, data, trgt, layers_id, overwrite=False):
        def reconstructRun(input, n_layer):
            return k_model.get_layer_n_output(n_layer, input)

        k_model = KerasModel(self.params)
        window_size = self.params.input_shape[0]
        for cv_name, cv in self.n_cv.cv.items():
            k_model.selectFoldConfig(10, mode=cv_name)
            for n, layer in enumerate(self.params.layers):
                if layer.identifier in layers_id:
                    layerpath = self.an_path + '/%s/outputs/layer_%s_%i/' % (cv_name, layer.identifier, n)

                    for i_fold, (_, test_index) in enumerate(cv):
                        k_model.load(k_model.model_best + '/%i_fold.h5' % i_fold)

                        x_test, y_test = lofar2image(data, trgt, test_index, window_size, window_size, self.n_cv.info)

                        for cls_i, cls_str in self.class_labels.items():
                            run = x_test[y_test == cls_i]

                            # TODO pass this to NestedCV and recover here as an attribute
                            ship_name = [ship_name
                                         for ship_name, ship_indices in self.n_cv.info.runs_named[cls_str].items()
                                         if np.isin(test_index, ship_indices).any()]

                            output_run = np.concatenate(reconstructRun(run, n), axis=0)
                            for ch_i, channel in enumerate(np.rollaxis(output_run, 2)):
                                outpath = layerpath + '/%s/channel_%i/' % (cls_str, ch_i)

                                if not exists(outpath):
                                    mkdir(outpath)
                                else:
                                    pass
                                    # if not overwrite:
                                    #     print "Layer %s_%i output already plotted. For overwriting current plots, " \
                                    #           "set overwrite to True. Exiting." % (layer.identifier, n)
                                    #     break

                                plotLOFARgram(channel, filename=outpath + '/%s.pdf' % ship_name)







    def _reconstructPredictions(self, data, trgt, image_window):
        """Retro-compatibility function. Used to update prediction files old storage format to the new one"""
        class_labels = self.class_labels
        run_info = SonarRunsInfo(self.n_cv.audiodatapath)
        for cv_name, cv in self.n_cv.cv.items():
            gt = np.array([])
            fold_path = self.modelpath + '/%s' % cv_name
            predictions = pd.DataFrame(columns=class_labels.values())
            prediction_pd = pd.DataFrame(columns=np.concatenate([class_labels.values(), ['Label']]))
            for i_fold, (train_index, test_index) in enumerate(cv):
                x_test, fold_trgt = lofar2image(data, trgt, test_index,
                                                image_window, image_window, run_indices_info=run_info)

                model = load_model(fold_path + '/best_states/' + '/%i_fold.h5' % i_fold)
                prediction = model.predict(x_test)

                del model

                prediction = pd.DataFrame(prediction, columns=np.concatenate([class_labels.values()]),
                                          index=pd.MultiIndex.from_product(
                                              [['fold_%i' % int(i_fold)], range(prediction.shape[0])]))
                prediction['Label'] = np.array(fold_trgt, dtype=int)

                prediction_pd = pd.concat([prediction_pd, prediction], axis=0)

                gt = np.concatenate([gt, fold_trgt], axis=0)

                gc.collect()
            preds = prediction_pd.reindex(pd.MultiIndex.from_tuples(prediction_pd.index.values))
            preds.to_csv(fold_path + '/pred.csv')

    def _renameIndexLevels(self):
        """Retro-compatibility function. IndexLevels will be renamed soon"""
        raise NotImplementedError