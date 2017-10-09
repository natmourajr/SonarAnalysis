"""
Author: Vinícius dos Santos Mello <viniciusdsmello@poli.ufrj.br>
Class created to implement a Stacked Auto Encoder for Classification and Novelty Detection.
"""
import os
import pickle
import numpy as np
import time

from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import metrics

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam, SGD
import keras.callbacks as callbacks
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

import multiprocessing

from Functions import TrainParameters as trnparams

num_process = multiprocessing.cpu_count()

class StackedAutoEncoder:
    def __init__(self, trn_params = None, development_flag = False, n_folds = 2, n_inits = 1, save_path='', prefix_str='RawData'):
        self.trn_params       = trn_params
        self.development_flag = development_flag
        self.n_folds          = n_folds
        self.n_inits          = n_inits
        self.save_path        = save_path
        self.prefix_str       = prefix_str


        self.n_inits          = self.trn_params.params['n_inits']
        self.params_str       = self.trn_params.get_params_str()
        self.analysis_str     = 'StackedAutoEncoder'
    '''
        Method that creates a string in the format: (InputDimension)x(1º Layer Dimension)x...x(Nº Layer Dimension)
    '''
    def getNeuronsString(self, hidden_neurons=[]):
        neurons_str = str(data.shape[1])
        for ineuron in hidden_neurons:
            neurons_str = neurons_str + 'x' + str(ineuron)
        neurons_str = neurons_str + 'x' + str(n_neurons)
        return neurons_str
    '''
        Method that preprocess data normalizing it according to parameter norm.
    '''
    def normalizeData(self, data, trgt):
        #normalize data based in train set
        if self.trn_params.params['norm'] == 'mapstd':
            scaler = preprocessing.StandardScaler().fit(data[train_id,:])
        elif self.trn_params.params['norm'] == 'mapstd_rob':
            scaler = preprocessing.RobustScaler().fit(data[train_id,:])
        elif self.trn_params.params['norm'] == 'mapminmax':
            scaler = preprocessing.MinMaxScaler().fit(data[train_id,:])
        norm_data = scaler.transform(data)
        norm_trgt = norm_data
        return norm_data, norm_trgt
    '''
        Method used to perform the layerwise algorithm to train the SAE
    '''
    def trainLayer(self, data=None, trgt=None, ifold=0, hidden_neurons = [400], layer=1):
        # Change elements equal to zero to one
        for i in range(len(hidden_neurons)):
            if hidden_neurons[i] == 0:
                hidden_neurons[i] = 1
        # load or create cross validation ids
        CVO = trnparams.ClassificationFolds(folder=save_path,
                                            n_folds=self.n_folds,
                                            trgt=trgt,
                                            dev=self.development_flag)

        neurons_str = self.getNeuronsString(hidden_neurons[:layer])

        model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(self.save_path, self.analysis_str,
                                                       self.prefix_str, n_folds,
                                                       params_str, neurons_str)
        if not self.development_flag:
            file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if self.trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                return 0
        else:
            file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if self.trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                return 0

        train_id, test_id = CVO[ifold]

        norm_data, norm_trgt = self.normalizeData(data, trgt)

        best_init = 0
        best_loss = 999

        classifier = []
        trn_desc = {}

        for i_init in range(n_inits):
            print 'Neuron: %i - Fold %i of %i Folds -  Init %i of %i Inits'%(hidden_neurons[layer-1],
                                                                             ifold+1,
                                                                             self.n_folds,
                                                                             i_init+1,
                                                                             n_inits)
            if layer == 1:
                model = Sequential()
                model.add(Dense(hidden_neurons[layer-1], input_dim=data.shape[1], init="uniform"))
                model.add(Activation(trn_params.params['hidden_activation']))
                model.add(Dense(data.shape[1], init="uniform"))
                model.add(Activation(trn_params.params['output_activation']))
            else if layer > 1:
                ilayer = 1
                while(ilayer <= layer):
                    neurons_str = self.getNeuronsString(hidden_neurons[:ilayer])
                    previous_model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(self.save_path,
                                                                            self.analysis_str,
                                                                            self.prefix_str,
                                                                            self.n_folds,
                                                                            params_str,
                                                                            neurons_str)
                    if not dev:
                        file_name = '%s_fold_%i_model.h5'%(previous_model_str,ifold)
                    else:
                        file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str,ifold)

                    layer_model = load_model(file_name)

                    get_layer_output = K.function([layer_model.layers[0].input],
                                                  [layer_model.layers[1].output])
                    if(ilayer == 1):
                        proj_all_data = norm_data
                    # Projection of layer
                    proj_all_data = get_layer_output([proj_all_data])[0]
                model = Sequential()
                model.add(Dense(hidden_neurons[layer], input_dim=proj_all_data.shape[1], init="uniform"))
                model.add(Activation(self.trn_params.params['hidden_activation']))
                model.add(Dense(proj_all_data.shape[1], init="uniform"))
                model.add(Activation(self.trn_params.params['output_activation']))
                norm_data = proj_all_data
            else:
                print "Error: Invalid number of layer. It must be equal or greater than 1"

            # Optimizer
            adam = Adam(lr=self.trn_params.params['learning_rate'],
                        beta_1=self.trn_params.params['beta_1'],
                        beta_2=self.trn_params.params['beta_2'],
                        epsilon=self.trn_params.params['epsilon'])

            model.compile(loss='mean_squared_error',
                          optimizer=adam,
                          metrics=['accuracy'])

            # Train model
            earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=self.trn_params.params['patience'],
                                                    verbose=self.trn_params.params['train_verbose'],
                                                    mode='auto')

            init_trn_desc = model.fit(norm_data[train_id], norm_data[train_id],
                                      nb_epoch=self.trn_params.params['n_epochs'],
                                      batch_size=self.trn_params.params['batch_size'],
                                      callbacks=[earlyStopping],
                                      verbose=self.trn_params.params['verbose'],
                                      validation_data=(norm_data[test_id],
                                                       norm_data[test_id]),
                                      shuffle=True)
            if np.min(init_trn_desc.history['val_loss']) < best_loss:
                best_init = i_init
                best_loss = np.min(init_trn_desc.history['val_loss'])
                classifier = model
                trn_desc['epochs'] = init_trn_desc.epoch
                trn_desc['acc'] = init_trn_desc.history['acc']
                trn_desc['loss'] = init_trn_desc.history['loss']
                trn_desc['val_loss'] = init_trn_desc.history['val_loss']
                trn_desc['val_acc'] = init_trn_desc.history['val_acc']

        # save model
        if not self.development_flag:
            file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
            classifier.save(file_name)
            file_name = '%s_fold_%i_trn_desc.jbl'%(model_str,ifold)
            joblib.dump([trn_desc],file_name,compress=9)
        else:
            file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
            classifier.save(file_name)
            file_name = '%s_fold_%i_trn_desc_dev.jbl'%(model_str,ifold)
            joblib.dump([trn_desc],file_name,compress=9)

    '''
        Function used to do a Fine Tuning in Stacked Auto Encoder for Classification of the data
        hidden_neurons contains the number of neurons in the sequence: [FirstLayer, SecondLayer, ... ]
    '''
    def trainClassifier(data=None, trgt=None, ifold=0, hidden_neurons=[400], layer=1):
        for i in range(len(hidden_neurons)):
            if hidden_neurons[i] == 0:
                hidden_neurons[i] = 1

        # Turn trgt to one-hot encoding
        trgt_sparse = np_utils.to_categorical(trgt.astype(int))

        # load or create cross validation ids
        CVO = self.trnparams.ClassificationFolds(folder=self.save_path,
                                                 n_folds=self.n_folds,
                                                 trgt=trgt,
                                                 dev=self.development_flag)

        neurons_str = self.getNeuronsString(hidden_neurons[:layer]) + 'x' + str(trgt_sparse.shape[1])
        model_str = '%s/%s/Classification_(%s)_%s_%i_folds_%s'%(self.save_path,
                                                                self.analysis_str,
                                                                neurons_str,
                                                                self.prefix_str,
                                                                self.n_folds,
                                                                self.params_str)
        if not self.development_flag:
            file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if self.trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                return 0
        else:
            file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if self.trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                return 0

        train_id, test_id = CVO[ifold]

        norm_data, norm_trgt = self.normalizeData(data, trgt)

        best_init = 0
        best_loss = 999

        classifier = []
        trn_desc = {}

        for i_init in range(self.n_inits):
            print 'Fold %i of %i Folds -  Init %i of %i Inits'%(ifold+1,
                                                                self.n_folds,
                                                                i_init+1,
                                                                self.n_inits)
            # Start the model
            model = Sequential()
            # Add layers
            ilayer = 1
            while(ilayer <= layer):
                 # Get the weights of ilayer
                neurons_str = self.getNeuronsString(hidden_neurons[:ilayer])
                previous_model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(self.save_path,
                                                                        self.analysis_str,
                                                                        self.prefix_str,
                                                                        self.n_folds,
                                                                        self.params_str,
                                                                        neurons_str)

                if not self.development_flag:
                    file_name = '%s_fold_%i_model.h5'%(previous_model_str,ifold)
                else:
                    file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str,ifold)

                # Check if the layer was trained
                if not os.path.exists(file_name):
                    def trainFold(ifold):
                        return self.trainLayer(data=data,
                                                trgt=data,
                                                ifold=ifold,
                                                layer=ilayer,
                                                hidden_neurons = hidden_neurons[:ilayer])

                    p = Pool(processes=num_processes)
                    folds = range(len(CVO))
                    results = p.map(trainFold, folds)
                    p.close()
                    p.join()

                layer_model = load_model(file_name)
                layer_weights = layer_model.layers[0].get_weights()
                if ilayer == 1:
                    model.add(Dense(hidden_neurons[ilayer-1], input_dim=norm_data.shape[1], weights=layer_weights,
                                    trainable=True))
                else:
                    model.add(Dense(hidden_neurons[ilayer-1], weights=layer_weights, trainable=True))
                model.add(Activation(trn_params.params['hidden_activation']))

            # Add Output Layer
            model.add(Dense(trgt_sparse.shape[1], init="uniform"))
            model.add(Activation('softmax'))

            adam = Adam(lr=self.trn_params.params['learning_rate'],
                        beta_1=self.trn_params.params['beta_1'],
                        beta_2=self.trn_params.params['beta_2'],
                        epsilon=self.trn_params.params['epsilon'])

            model.compile(loss='mean_squared_error',
                          optimizer=adam,
                          metrics=['accuracy'])
            # Train model
            earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=self.trn_params.params['patience'],
                                                    verbose=self.trn_params.params['train_verbose'],
                                                    mode='auto')

            init_trn_desc = model.fit(norm_data[train_id], trgt_sparse[train_id],
                                      nb_epoch=self.trn_params.params['n_epochs'],
                                      batch_size=self.trn_params.params['batch_size'],
                                      callbacks=[earlyStopping],
                                      verbose=self.trn_params.params['verbose'],
                                      validation_data=(norm_data[test_id],
                                                       trgt_sparse[test_id]),
                                      shuffle=True)
            if np.min(init_trn_desc.history['val_loss']) < best_loss:
                best_init = i_init
                best_loss = np.min(init_trn_desc.history['val_loss'])
                classifier = model
                trn_desc['epochs'] = init_trn_desc.epoch
                trn_desc['acc'] = init_trn_desc.history['acc']
                trn_desc['loss'] = init_trn_desc.history['loss']
                trn_desc['val_loss'] = init_trn_desc.history['val_loss']
                trn_desc['val_acc'] = init_trn_desc.history['val_acc']

        # save model
        if not self.development_flag:
            file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
            classifier.save(file_name)
            file_name = '%s_fold_%i_trn_desc.jbl'%(model_str,ifold)
            joblib.dump([trn_desc],file_name,compress=9)
        else:
            file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
            classifier.save(file_name)
            file_name = '%s_fold_%i_trn_desc_dev.jbl'%(model_str,ifold)
            print file_name
            joblib.dump([trn_desc],file_name,compress=9)