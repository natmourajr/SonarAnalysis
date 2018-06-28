# -*- coding: utf-8 -*-
'''
   Author: Vinícius dos Santos Mello viniciusdsmello at poli.ufrj.br
   Class created to implement a Stacked Autoencoder for Classification and Novelty Detection.
'''
import os
import pickle
import numpy as np
import time

from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import metrics

from keras.models import Sequential
from keras import regularizers
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam, SGD
import keras.callbacks as callbacks
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from keras import losses

from Functions import TrainParameters as trnparams
from Functions.MetricsLosses import kullback_leibler_divergence

import multiprocessing

num_processes = multiprocessing.cpu_count()

class StackedAutoEncoders:
    def __init__(self, params = None, development_flag = False, n_folds = 2, save_path='', prefix_str='RawData', CVO=None,
                 noveltyDetection=False, inovelty = 0, allow_change_weights=False):
        self.trn_params       = params
        self.development_flag = development_flag
        self.n_folds          = n_folds
        self.save_path        = save_path
        self.noveltyDetection = noveltyDetection
        self.allow_change_weights = allow_change_weights
        self.inovelty = inovelty

        self.n_inits          = self.trn_params.params['n_inits']
        self.params_str       = self.trn_params.get_params_str()
        self.analysis_str     = 'StackedAutoEncoder'

        # Distinguish between a SAE for Novelty Detection and SAE for 'simple' Classification
        if noveltyDetection:
            self.CVO          = CVO[inovelty]
            self.prefix_str   = prefix_str+'_%i_novelty'%(inovelty)
        else:
            self.CVO          = CVO
            self.prefix_str   = prefix_str

        # Choose optmizer algorithm
        if self.trn_params.params['optmizerAlgorithm'] == 'SGD':
            self.optmizer = SGD(lr=self.trn_params.params['learning_rate'],
                                    # momentum=self.trn_params.params['momentum'],
                                    # decay=self.trn_params.params['decay'],
                                    nesterov=self.trn_params.params['nesterov'])

        elif self.trn_params.params['optmizerAlgorithm'] == 'Adam':
            self.optmizer = Adam(lr=self.trn_params.params['learning_rate'],
                                    beta_1=self.trn_params.params['beta_1'],
                                    beta_2=self.trn_params.params['beta_2'],
                                    epsilon=self.trn_params.params['epsilon'])
        else:
            self.optmizer = self.trn_params.params['optmizerAlgorithm']

        # Choose loss functions
        if self.trn_params.params['loss'] == 'kullback_leibler_divergence':
            self.lossFunction = kullback_leibler_divergence
        else:
            self.lossFunction = self.trn_params.params['loss']
        losses.custom_loss = self.lossFunction
    '''
        Method that creates a string in the format: (InputDimension)x(1º Layer Dimension)x...x(Nº Layer Dimension)
    '''
    def getNeuronsString(self, data, hidden_neurons=[]):
        neurons_str = str(data.shape[1])
        for ineuron in hidden_neurons:
            neurons_str = neurons_str + 'x' + str(ineuron)
        return neurons_str
    '''
        Method that preprocess data normalizing it according to 'norm' parameter.
    '''
    def normalizeData(self, data, ifold):
        #normalize data based in train set
        train_id, test_id = self.CVO[ifold]
        if self.trn_params.params['norm'] == 'mapstd':
            scaler = preprocessing.StandardScaler().fit(data[train_id,:])
        elif self.trn_params.params['norm'] == 'mapstd_rob':
            scaler = preprocessing.RobustScaler().fit(data[train_id,:])
        elif self.trn_params.params['norm'] == 'mapminmax':
            scaler = preprocessing.MinMaxScaler().fit(data[train_id,:])
        else: 
            return data
        norm_data = scaler.transform(data)

        return norm_data

    '''
        Method that return the Stacked AutoEncoder model
    '''
    def getModel(self, data, trgt, hidden_neurons=[1], layer=1, ifold=0, regularizer=None, regularizer_param=None):
        if layer > len(hidden_neurons):
            print "[-] Error: The parameter layer must be less or equal to the size of list hidden_neurons"
            return 1
        if layer == 1:
            neurons_str = self.getNeuronsString(data, hidden_neurons[:layer])
            if regularizer != None and len(regularizer) != 0:
                previous_model_str = os.path.join(self.save_path,
                                                  "saeModel_%i_noveltyID_%s_neurons_%s_regularizer(%.3f)"%(self.inovelty, neurons_str, regularizer, regularizer_param)
                                                 )
                
            else:
                previous_model_str = os.path.join(self.save_path,"saeModel_%i_noveltyID_%s_neurons"%(self.inovelty, neurons_str))
                
            if not self.development_flag:
                file_name = '%s_fold_%i_model.h5'%(previous_model_str,ifold)
            else:
                file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str,ifold)

            # Check if previous layer model was trained
            if not os.path.exists(file_name):
                self.trainLayer(data=data,
                                trgt=trgt,
                                ifold=ifold,
                                hidden_neurons = hidden_neurons[:layer],
                                layer=layer,
                                regularizer=regularizer,
                                regularizer_param=regularizer_param)

            model = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
        elif layer > 1:
            layer_models = {}
            layer_encoder_weights = {}
            layer_decoder_weights = {}

            for ilayer in range(1,layer+1):
                neurons_str = self.getNeuronsString(data, hidden_neurons[:ilayer])
                if regularizer != None and len(regularizer) != 0:
                    
                    previous_model_str = os.path.join(self.save_path,
                                                      "saeModel_%i_noveltyID_%s_neurons_%s_regularizer(%.3f)"%(self.inovelty, neurons_str, regularizer, regularizer_param)
                                                     )
                else:
                    previous_model_str = os.path.join(self.save_path,"saeModel_%i_noveltyID_%s_neurons"%(self.inovelty, neurons_str))
                if not self.development_flag:
                    file_name = '%s_fold_%i_model.h5'%(previous_model_str,ifold)
                else:
                    file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str,ifold)

                # Check if previous layer model was trained
                if not os.path.exists(file_name):
                    self.trainLayer(data=data,
                                    trgt=trgt,
                                     ifold=ifold,
                                      hidden_neurons = hidden_neurons[:ilayer],
                                      layer=ilayer,
                                      regularizer=regularizer,
                                      regularizer_param=regularizer_param)

                layer_models[ilayer] = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})

                layer_encoder_weights[ilayer] = layer_models[ilayer].layers[0].get_weights()
                layer_decoder_weights[ilayer] = layer_models[ilayer].layers[2].get_weights()

            model = Sequential()
            # Encoder
            for ilayer in range(1,layer+1):
                if self.development_flag:
                    print '[*] Adding Encoder of layer %i (%i neurons)'%(ilayer,hidden_neurons[ilayer-1])
                if ilayer == 1:
                    model.add(Dense(hidden_neurons[ilayer-1], input_dim=data.shape[1], weights=layer_encoder_weights[ilayer], trainable=False))
                    model.add(Activation(self.trn_params.params['hidden_activation']))
                else:
                    model.add(Dense(hidden_neurons[ilayer-1], weights=layer_encoder_weights[ilayer], trainable=False))
                    model.add(Activation(self.trn_params.params['hidden_activation']))
            # Decoder
            for ilayer in range(layer-1,0,-1):
                if self.development_flag:
                    print '[*] Adding Decoder of layer %i (%i neurons)'%(ilayer,hidden_neurons[ilayer-1])
                model.add(Dense(hidden_neurons[ilayer-1], weights=layer_decoder_weights[ilayer+1], trainable=False))
                model.add(Activation(self.trn_params.params['output_activation']))

            model.add(Dense(data.shape[1], weights=layer_decoder_weights[ilayer], trainable=False))
            model.add(Activation(self.trn_params.params['output_activation']))

        return model

    '''
        Method that returns the encoder of an intermediate layer.
    '''
    def getEncoder(self, data, trgt, hidden_neurons=[1], layer=1, ifold=0, regularizer=None, regularizer_param=None):
        for i in range(len(hidden_neurons)):
            if hidden_neurons[i] == 0:
                hidden_neurons[i] = 1

        if (layer <= 0) or (layer > len(hidden_neurons)):
            print "[-] Error: The parameter layer must be greater than zero and less or equal to the length of list hidden_neurons"
            return -1

        # Turn trgt to one-hot encoding
        trgt_sparse = np_utils.to_categorical(trgt.astype(int))

        neurons_str = self.getNeuronsString(data,hidden_neurons[:layer]) + 'x' + str(trgt_sparse.shape[1])

        train_id, test_id = self.CVO[ifold]

        norm_data = self.normalizeData(data, ifold)

        best_init = 0
        best_loss = 999

        # Start the model
        model = Sequential()
        # Add layers
        for ilayer in range(1,layer+1):
             # Get the weights of ilayer
            
            neurons_str = self.getNeuronsString(data, hidden_neurons[:layer])
            if regularizer != None and len(regularizer) != 0:
                previous_model_str = os.path.join(self.save_path,
                                                  "saeModel_%i_noveltyID_%s_neurons_%s_regularizer(%.3f)"%(self.inovelty, neurons_str, regularizer, regularizer_param)
                                                 )
                
            else:
                previous_model_str = os.path.join(self.save_path,"saeModel_%i_noveltyID_%s_neurons"%(self.inovelty, neurons_str))
                
            if not self.development_flag:
                file_name = '%s_fold_%i_model.h5'%(previous_model_str,ifold)
            else:
                file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str,ifold)
                
            # Check if the layer was trained
            if not os.path.exists(file_name):
                self.trainLayer(data=data,
                                trgt=data,
                                ifold=ifold,
                                layer=ilayer,
                                hidden_neurons = hidden_neurons[:ilayer],
                                regularizer=regularizer,
                                regularizer_param=regularizer_param)

            layer_model = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
            layer_weights = layer_model.layers[0].get_weights()
            if ilayer == 1:
                model.add(Dense(units=hidden_neurons[0], input_dim=norm_data.shape[1], weights=layer_weights,
                                trainable=True))
            else:
                model.add(Dense(units=hidden_neurons[ilayer-1], weights=layer_weights, trainable=True))

            model.add(Activation(self.trn_params.params['hidden_activation']))

        return model
    

    '''
        Method used to perform the layerwise algorithm to train the SAE
    ''' 
    def trainLayer(self, data=None, trgt=None, ifold=0, hidden_neurons = [400], layer=1, regularizer=None, regularizer_param=None):
        # Change elements equal to zero to one
        for i in range(len(hidden_neurons)):
            if hidden_neurons[i] == 0:
                hidden_neurons[i] = 1
        if (layer <= 0) or (layer > len(hidden_neurons)):
            print "[-] Error: The parameter layer must be greater than zero and less or equal to the length of list hidden_neurons"
            return -1

        if self.trn_params.params['verbose']:
            print '[+] Using %s as optmizer algorithm'%self.trn_params.params['optmizerAlgorithm']

        
        neurons_str = self.getNeuronsString(data, hidden_neurons[:layer])
        if regularizer != None and len(regularizer) != 0:
            model_str = os.path.join(self.save_path,
                                              "saeModel_%i_noveltyID_%s_neurons_%s_regularizer(%.3f)"%(self.inovelty, neurons_str, regularizer, regularizer_param)
                                    )

        else:
            model_str = os.path.join(self.save_path,"saeModel_%i_noveltyID_%s_neurons"%(self.inovelty, neurons_str))
        
        if not self.development_flag:
            file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if self.trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                # load model
                file_name  = '%s_fold_%i_model.h5'%(model_str,ifold)
                classifier = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
                file_name  = '%s_fold_%i_trn_desc.jbl'%(model_str,ifold)
                trn_desc   = joblib.load(file_name)
                return ifold, classifier, trn_desc
        else:
            file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if self.trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                # load model
                file_name  = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
                classifier = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
                file_name  = '%s_fold_%i_trn_desc_dev.jbl'%(model_str,ifold)
                trn_desc   = joblib.load(file_name)
                return ifold, classifier, trn_desc

        train_id, test_id = self.CVO[ifold]

        norm_data = self.normalizeData(data, ifold)

        best_init = 0
        best_loss = 999

        classifier = []
        trn_desc = {}

        for i_init in range(self.n_inits):
            print 'Autoencoder - Layer: %i - Topology: %s - Fold %i of %i Folds -  Init %i of %i Inits'%(layer,
                                                                                                        neurons_str,
                                                                                                        ifold+1,
                                                                                                        self.n_folds,
                                                                                                        i_init+1,
                                                                                                        self.n_inits)
            model = Sequential()
            proj_all_data = norm_data
            if layer == 1:

                model.add(Dense(units=hidden_neurons[layer-1], input_dim=data.shape[1], kernel_initializer="uniform"))
                model.add(Activation(self.trn_params.params['hidden_activation']))
                model.add(Dense(units=data.shape[1], input_dim=hidden_neurons[layer-1], kernel_initializer="uniform"))
                model.add(Activation(self.trn_params.params['output_activation']))
    	    elif layer > 1:
                for ilayer in range(1,layer):
                    neurons_str = self.getNeuronsString(data, hidden_neurons[:ilayer])
                    if regularizer != None and len(regularizer) != 0:
                        previous_model_str = os.path.join(self.save_path,
                                                          "saeModel_%i_noveltyID_%s_neurons_%s_regularizer(%.3f)"%(self.inovelty, neurons_str, regularizer, regularizer_param)
                                                         )

                    else:
                        previous_model_str = os.path.join(self.save_path, "saeModel_%i_noveltyID_%s_neurons"%(self.inovelty, neurons_str))
                        
                    if not self.development_flag:
                        file_name = '%s_fold_%i_model.h5'%(previous_model_str,ifold)
                    else:
                        file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str,ifold)

                    # Check if previous layer model was trained
                    if not os.path.exists(file_name):
                        self.trainLayer(data=data,
                                        trgt=trgt,
                                        ifold=ifold,
                                        hidden_neurons = hidden_neurons[:ilayer],
                                        layer=ilayer,
                                        regularizer=regularizer,
                                        regularizer_param=regularizer_param)

                    layer_model = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})

                    get_layer_output = K.function([layer_model.layers[0].input],
                                                  [layer_model.layers[1].output])
                    # Projection of layer
                    proj_all_data = get_layer_output([proj_all_data])[0]

                model.add(Dense(units=hidden_neurons[layer-1], input_dim=proj_all_data.shape[1], kernel_initializer="uniform"))
                model.add(Activation(self.trn_params.params['hidden_activation']))
                if regularizer == "dropout":
                    model.add(Dropout(regularizer_param))
                elif regularizer == "l1":
                    model.add(Dense(units=proj_all_data.shape[1], input_dim=hidden_neurons[layer-1],
                                    kernel_initializer="uniform", kernel_regularizer=regularizers.l1(regularizer_param)))
                elif regularizer == "l2":
                    model.add(Dense(units=proj_all_data.shape[1], input_dim=hidden_neurons[layer-1],
                                    kernel_initializer="uniform", kernel_regularizer=regularizers.l2(regularizer_param)))
                else:
                    model.add(Dense(units=proj_all_data.shape[1], input_dim=hidden_neurons[layer-1],kernel_initializer="uniform"))
                model.add(Activation(self.trn_params.params['output_activation']))
                norm_data = proj_all_data
    	    # end of elif layer > 1:

            model.compile(loss=self.lossFunction,
                          optimizer=self.optmizer,
                          metrics=self.trn_params.params['metrics'])
            # Train model
            earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=self.trn_params.params['patience'],
                                                    verbose=self.trn_params.params['train_verbose'],
                                                    mode='auto')

            init_trn_desc = model.fit(norm_data[train_id], norm_data[train_id],
                                      epochs=self.trn_params.params['n_epochs'],
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

                for imetric in range(len(self.trn_params.params['metrics'])):
                    if self.trn_params.params['metrics'][imetric] == 'accuracy':
                        metric = 'acc'
                    else:
                        metric = self.trn_params.params['metrics'][imetric]
                    trn_desc[metric] = init_trn_desc.history[metric]
                    trn_desc['val_'+metric] = init_trn_desc.history['val_'+metric]

                trn_desc['loss'] = init_trn_desc.history['loss']
                trn_desc['val_loss'] = init_trn_desc.history['val_loss']

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
        return ifold, classifier, trn_desc

    '''
        Method that return the classifier according to topology parsed
    '''
    def loadClassifier(self, data=None, trgt=None, hidden_neurons=[1], layer=1, ifold=0, regularizer=None, regularizer_param=None):
        for i in range(len(hidden_neurons)):
            if hidden_neurons[i] == 0:
                hidden_neurons[i] = 1
        if (layer <= 0) or (layer > len(hidden_neurons)):
            print "[-] Error: The parameter layer must be greater than zero and less or equal to the length of list hidden_neurons"
            return -1

        # Turn trgt to one-hot encoding
        trgt_sparse = np_utils.to_categorical(trgt.astype(int))

        # load model
        neurons_str = self.getNeuronsString(data,hidden_neurons[:layer]) + 'x' + str(trgt_sparse.shape[1])

        if regularizer != None and len(regularizer) != 0:
            previous_model_str = os.path.join(self.save_path,
                                              "classifierModel_%i_noveltyID_%s_neurons_%s_regularizer(%.3f)"%(self.inovelty, neurons_str, regularizer, regularizer_param)
                                             )

        else:
            previous_model_str = os.path.join(self.save_path,"classifierModel_%i_noveltyID_%s_neurons"%(self.inovelty, neurons_str))

        classifier = {}
        if not self.development_flag:
            file_name  = '%s_fold_%i_model.h5'%(model_str,ifold)
            try:
                classifier = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
            except:
                print '[-] Error: File or Directory not found'
                return
        else:
            file_name  = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
            try:
                classifier = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
            except:
                print '[-] Error: File or Directory not found'
        return classifier

    '''
        Function used to do a Fine Tuning in Stacked Auto Encoder for Classification of the data
        hidden_neurons contains the number of neurons in the sequence: [FirstLayer, SecondLayer, ... ]
    '''
    def trainClassifier(self, data=None, trgt=None, ifold=0, hidden_neurons=[1], layer=1, regularizer=None, regularizer_param=None):
        for i in range(len(hidden_neurons)):
            if hidden_neurons[i] == 0:
                hidden_neurons[i] = 1

        if (layer <= 0) or (layer > len(hidden_neurons)):
            print "[-] Error: The parameter layer must be greater than zero and less or equal to the length of list hidden_neurons"
            return -1

        # Turn trgt to one-hot encoding
        trgt_sparse = np_utils.to_categorical(trgt.astype(int))

        neurons_str = self.getNeuronsString(data,hidden_neurons[:layer])

        if regularizer != None and len(regularizer) != 0:
            model_str = os.path.join(self.save_path,
                                              "classifierModel_%i_noveltyID_%s_neurons_%s_regularizer(%.3f)"%(self.inovelty, neurons_str, regularizer, regularizer_param)
                                             )

        else:
            model_str = os.path.join(self.save_path,"classifierModel_%i_noveltyID_%s_neurons"%(self.inovelty, neurons_str))
                                                          
        if not self.development_flag:
            file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if self.trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                # load model
                file_name  = '%s_fold_%i_model.h5'%(model_str,ifold)
                classifier = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
                file_name  = '%s_fold_%i_trn_desc.jbl'%(model_str,ifold)
                trn_desc   = joblib.load(file_name)
                return ifold, classifier, trn_desc
        else:
            file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if self.trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                # load model
                file_name  = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
                classifier = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
                file_name  = '%s_fold_%i_trn_desc_dev.jbl'%(model_str,ifold)
                trn_desc   = joblib.load(file_name)
                return ifold, classifier, trn_desc

        train_id, test_id = self.CVO[ifold]

        norm_data = self.normalizeData(data, ifold)

        best_init = 0
        best_loss = 999

        classifier = []
        trn_desc = {}

        for i_init in range(self.n_inits):
            print 'Classifier - Layer: %i - Topology: %s - Fold: %i of %i Folds -  Init: %i of %i Inits'%(layer,
                                                                                                          neurons_str,
                                                                                                          ifold+1,
                                                                                                          self.n_folds,
                                                                                                          i_init+1,
                                                                                                          self.n_inits)
            # Start the model
            model = Sequential()
            # Add layers
            for ilayer in range(1,layer+1):
                 # Get the weights of ilayer
                neurons_str = self.getNeuronsString(data, hidden_neurons[:ilayer])
                if regularizer != None and len(regularizer) != 0:
                    previous_model_str = os.path.join(self.save_path,
                                                      "saeModel_%i_noveltyID_%s_neurons_%s_regularizer(%.3f)"%(self.inovelty, neurons_str, regularizer, regularizer_param)
                                                     )

                else:
                    previous_model_str = os.path.join(self.save_path,"saeModel_%i_noveltyID_%s_neurons"%(self.inovelty, neurons_str))

                if not self.development_flag:
                    file_name = '%s_fold_%i_model.h5'%(previous_model_str,ifold)
                else:
                    file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str,ifold)

                # Check if the layer was trained
                if not os.path.exists(file_name):
                    self.trainLayer(data=data,
                                    trgt=data,
                                    ifold=ifold,
                                    layer=ilayer,
                                    hidden_neurons = hidden_neurons[:ilayer],
                                    regularizer=regularizer,
                                    regularizer_param=regularizer_param)

                layer_model = load_model(file_name, custom_objects={'%s'%self.trn_params.params['loss']: self.lossFunction})
                layer_weights = layer_model.layers[0].get_weights()
                if ilayer == 1:
                    model.add(Dense(units=hidden_neurons[0], input_dim=norm_data.shape[1], weights=layer_weights,
                                    trainable=self.allow_change_weights))
                else:
                    model.add(Dense(units=hidden_neurons[ilayer-1], weights=layer_weights, trainable=self.allow_change_weights))

                model.add(Activation(self.trn_params.params['hidden_activation']))

            # Add Output Layer
            model.add(Dense(units=trgt_sparse.shape[1], kernel_initializer="uniform"))
            model.add(Activation('softmax'))

            model.compile(loss=self.lossFunction,
                          optimizer=self.optmizer,
                          metrics=self.trn_params.params['metrics'])
            # Train model
            earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=self.trn_params.params['patience'],
                                                    verbose=self.trn_params.params['train_verbose'],
                                                    mode='auto')

            init_trn_desc = model.fit(norm_data[train_id], trgt_sparse[train_id],
                                      epochs=self.trn_params.params['n_epochs'],
                                      batch_size=self.trn_params.params['batch_size'],
                                      callbacks=[earlyStopping],
                                      verbose=self.trn_params.params['verbose'],
                                      validation_data=(norm_data[test_id], trgt_sparse[test_id]),
                                      shuffle=True)
            if np.min(init_trn_desc.history['val_loss']) < best_loss:
                best_init = i_init
                best_loss = np.min(init_trn_desc.history['val_loss'])
                classifier = model
                trn_desc['epochs'] = init_trn_desc.epoch

                for imetric in range(len(self.trn_params.params['metrics'])):
                    if self.trn_params.params['metrics'][imetric] == 'accuracy':
                        metric = 'acc'
                    else:
                        metric = self.trn_params.params['metrics'][imetric]
                    trn_desc[metric] = init_trn_desc.history[metric]
                    trn_desc['val_'+metric] = init_trn_desc.history['val_'+metric]

                trn_desc['loss'] = init_trn_desc.history['loss']
                trn_desc['val_loss'] = init_trn_desc.history['val_loss']

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
        return ifold, classifier, trn_desc