"""
    Author: Vinicius dos Santos Mello 
            viniciusdsmello@poli.ufrj.br
    
    Description: This file contains functions used for Neural Network training.
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

'''
    Function used to do a Fine Tuning in Stacked Auto Encoder for Classification of the data
    hidden_neurons contains the number of neurons in the sequence: [FirstLayer, SecondLayer, ... ]
'''
def SAEClassificationTrainFunction(data=None, trgt=None, ifold=0, 
                                   n_folds=2, hidden_neurons=[],
                                   trn_params=None, save_path='', dev=False):
    if len(hidden_neurons) == 0:
        hidden_neurons = [1]
        
    for i in range(len(hidden_neurons)):
        if hidden_neurons[i] == 0:
            hidden_neurons[i] = 1
    
    trgt_sparse = np_utils.to_categorical(trgt.astype(int))
    # load or create cross validation ids
    CVO = trnparams.ClassificationFolds(folder=save_path,n_folds=n_folds,trgt=trgt,dev=dev)

    n_folds = len(CVO)
    n_inits = trn_params.params['n_inits']

    params_str = trn_params.get_params_str()
    
    analysis_str = 'StackedAutoEncoder'
    prefix_str = 'RawData'
    
    # Create a string like InputDimension x FirstLayerDimension x ... x OutputDimension
    neurons_str = str(data.shape[1])
    for ineuron in hidden_neurons:
        neurons_str = neurons_str + 'x' + str(ineuron)
    neurons_str = neurons_str + 'x' + str(trgt_sparse.shape[1])
    model_str = '%s/%s/Classification_(%s)_%s_%i_folds_%s'%(save_path,analysis_str,
                                                                     neurons_str,
                                                                     prefix_str, n_folds,
                                                                     params_str)

    
    if not dev:
        file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
        if os.path.exists(file_name):
            if trn_params.params['verbose']:
                print 'File %s exists'%(file_name)
            return 0
    else:
        file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
        if os.path.exists(file_name):
            if trn_params.params['verbose']:
                print 'File %s exists'%(file_name)
            return 0

    train_id, test_id = CVO[ifold]

    #normalize data based in train set
    if trn_params.params['norm'] == 'mapstd':
        scaler = preprocessing.StandardScaler().fit(data[train_id,:])
    elif trn_params.params['norm'] == 'mapstd_rob':
        scaler = preprocessing.RobustScaler().fit(data[train_id,:])
    elif trn_params.params['norm'] == 'mapminmax':
        scaler = preprocessing.MinMaxScaler().fit(data[train_id,:])

    norm_data = scaler.transform(data)
    
    best_init = 0
    best_loss = 999

    classifier = [] 
    trn_desc = {}

    for i_init in range(n_inits):
        print 'Fold %i of %i Folds -  Init %i of %i Inits'%(ifold+1, 
                                                            n_folds, 
                                                            i_init+1,
                                                            n_inits)
        
        # Get the weights of first layer
        layer = 1
        neurons_str = str(data.shape[1])
        for ineuron in hidden_neurons[:layer]:
            neurons_str = neurons_str + 'x' + str(ineuron)
        previous_model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(save_path,analysis_str,
                                                                prefix_str,
                                                                n_folds,
                                                                params_str,
                                                                neurons_str)

        if not dev:        
            file_name = '%s_fold_%i_model.h5'%(previous_model_str,ifold)
        else:
            file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str,ifold)
        if not os.path.exists(file_name):
            def trainFold(ifold):
                ineuron = hidden_neurons[1]
                return StackedAutoEncoderTrainFunction(data=all_data,
                                        trgt=all_data,
                                        ifold=ifold,
                                        n_folds=n_folds, 
                                        n_neurons=ineuron,
                                        trn_params=trn_params, 
                                        save_path=results_path,
                                        layer=layer,
                                        hidden_neurons = hidden_neurons[:layer],            
                                        dev=dev)

            p = Pool(processes=num_processes)
            folds = range(len(CVO))
            results = p.map(trainFold, folds)
            p.close()
            p.join() 

        first_layer_model = load_model(file_name)
        first_layer = first_layer_model.layers[0]
        first_layer_weights = first_layer.get_weights()
            
        model = Sequential()
        model.add(Dense(hidden_neurons[0], input_dim=norm_data.shape[1], weights=first_layer_weights, trainable=True))
        model.add(Activation(trn_params.params['hidden_activation']))
        # Add second hidden layer
        if(len(hidden_neurons) > 1):
            # Get the weights of second layer
            layer = 2
            neurons_str = str(data.shape[1])
            for ineuron in hidden_neurons[:layer]:
                neurons_str = neurons_str + 'x' + str(ineuron)
            previous_model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(save_path,analysis_str,
                                                                    prefix_str,
                                                                    n_folds,
                                                                    params_str,
                                                                    neurons_str)

            if not dev:        
                file_name = '%s_fold_%i_model.h5'%(previous_model_str,ifold)
            else:
                file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str,ifold)
            if not os.path.exists(file_name):
                def trainFold(ifold):
                    ineuron = hidden_neurons[1]
                    return StackedAutoEncoderTrainFunction(data=all_data,
                                            trgt=all_data,
                                            ifold=ifold,
                                            n_folds=n_folds, 
                                            n_neurons=ineuron,
                                            trn_params=trn_params, 
                                            save_path=results_path,
                                            layer=layer,
                                            hidden_neurons = hidden_neurons[:layer],              
                                            dev=dev)

                p = Pool(processes=num_processes)
                folds = range(len(CVO))
                results = p.map(trainFold, folds)
                p.close()
                p.join() 

            second_layer_model = load_model(file_name)
            second_layer = second_layer_model.layers[0]
            second_layer_weights = second_layer.get_weights()
            
            model.add(Dense(hidden_neurons[1], weights=second_layer_weights, trainable=True))
            model.add(Activation(trn_params.params['hidden_activation']))
        # Add third hidden layer    
        if(len(hidden_neurons) > 2):
             # Get the weights of third layer
            layer = 3
            neurons_str = str(data.shape[1])
            for ineuron in hidden_neurons[:layer]:
                neurons_str = neurons_str + 'x' + str(ineuron)
            previous_model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(save_path,analysis_str,
                                                                    prefix_str,
                                                                    n_folds,
                                                                    params_str,
                                                                    neurons_str)

            if not dev:        
                file_name = '%s_fold_%i_model.h5'%(previous_model_str,ifold)
            else:
                file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str,ifold)
            if not os.path.exists(file_name):
                def trainFold(ifold):
                    ineuron = hidden_neurons[1]
                    return StackedAutoEncoderTrainFunction(data=all_data,
                                            trgt=all_data,
                                            ifold=ifold,
                                            n_folds=n_folds, 
                                            n_neurons=ineuron,
                                            trn_params=trn_params, 
                                            save_path=results_path,
                                            layer=layer,
                                            hidden_neurons = hidden_neurons[:layer],              
                                            dev=dev)

                p = Pool(processes=num_processes)
                folds = range(len(CVO))
                results = p.map(trainFold, folds)
                p.close()
                p.join() 
            
            third_layer_model = load_model(file_name)
            third_layer = third_layer_model.layers[0]
            third_layer_weights = third_layer.get_weights()
            
            model.add(Dense(hidden_neurons[2], weights=third_layer_weights, trainable=True))
            model.add(Activation(trn_params.params['hidden_activation']))
            
        # Add Output Layer
        model.add(Dense(trgt_sparse.shape[1], init="uniform")) 
        model.add(Activation('softmax'))
        
        adam = Adam(lr=trn_params.params['learning_rate'], 
                    beta_1=trn_params.params['beta_1'],
                    beta_2=trn_params.params['beta_2'],
                    epsilon=trn_params.params['epsilon'])
        
        model.compile(loss='mean_squared_error', 
                      optimizer=adam,
                      metrics=['accuracy'])
        # Train model
        earlyStopping = callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=trn_params.params['patience'],
                                                verbose=trn_params.params['train_verbose'], 
                                                mode='auto')

        init_trn_desc = model.fit(norm_data[train_id], trgt_sparse[train_id], 
                                  nb_epoch=trn_params.params['n_epochs'], 
                                  batch_size=trn_params.params['batch_size'],
                                  callbacks=[earlyStopping], 
                                  verbose=trn_params.params['verbose'],
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
    if not dev:        
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
        
'''
    Function used to perform the layerwise algorithm to train the SAE
'''
def StackedAutoEncoderTrainFunction(data=None, trgt=None, 
                                    ifold=0, n_folds=2, n_neurons=10, 
                                    trn_params=None, save_path='', dev=False, layer = 1, hidden_neurons = [400]):
    
    if n_neurons == 0:
        n_neurons = 1
    
    # load or create cross validation ids
    CVO = trnparams.ClassificationFolds(folder=save_path,n_folds=n_folds,trgt=trgt,dev=dev)

    n_folds = len(CVO)
    n_inits = trn_params.params['n_inits']

    params_str = trn_params.get_params_str()
    
    analysis_str = 'StackedAutoEncoder'
    prefix_str = 'RawData'
    
    if layer == 1:
        neurons_str = str(data.shape[1]) + 'x' + str(n_neurons)
        model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(save_path,analysis_str,
                                                          prefix_str,n_folds,
                                                          params_str,neurons_str)


        if not dev:
            file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                return 0
        else:
            file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                return 0
    if layer == 2:
        neurons_str = str(data.shape[1])
        for ineuron in hidden_neurons[:layer]:
            neurons_str = neurons_str + 'x' + str(ineuron)
        neurons_str = neurons_str + 'x' + str(n_neurons)
        model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(save_path,analysis_str,
                                                                prefix_str,n_folds,
                                                                params_str,neurons_str)


        if not dev:
            file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                return 0
        else:
            file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                return 0
            
    train_id, test_id = CVO[ifold]

    #normalize data based in train set
    if trn_params.params['norm'] == 'mapstd':
        scaler = preprocessing.StandardScaler().fit(data[train_id,:])
    elif trn_params.params['norm'] == 'mapstd_rob':
        scaler = preprocessing.RobustScaler().fit(data[train_id,:])
    elif trn_params.params['norm'] == 'mapminmax':
        scaler = preprocessing.MinMaxScaler().fit(data[train_id,:])

    norm_data = scaler.transform(data)
    norm_trgt = norm_data
    
    best_init = 0
    best_loss = 999

    classifier = []
    trn_desc = {}
    
    for i_init in range(n_inits):
        print 'Neuron: %i - Fold %i of %i Folds -  Init %i of %i Inits'%(n_neurons, 
                                                                         ifold+1, 
                                                                         n_folds, 
                                                                         i_init+1,
                                                                         n_inits)
        if layer == 1:
            model = Sequential()
            model.add(Dense(n_neurons, input_dim=data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['hidden_activation']))
            model.add(Dense(data.shape[1], init="uniform")) 
            model.add(Activation(trn_params.params['output_activation']))
        if layer == 2:
            # Get the projection of the data from previous layer
            neurons_str = str(data.shape[1])
            for ineuron in hidden_neurons[:layer-1]:
                neurons_str = neurons_str + 'x' + str(ineuron)
            previous_model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(save_path,analysis_str,
                                                                    prefix_str,
                                                                    n_folds,
                                                                    params_str,
                                                                    neurons_str)
            if not dev:
                file_name = '%s_fold_%i_model.h5'%(previous_model_str,ifold)
            else:
                file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str,ifold)

            first_layer_model = load_model(file_name)

            get_layer_output = K.function([first_layer_model.layers[0].input],
                                          [first_layer_model.layers[1].output])
            proj_all_data = get_layer_output([norm_data])[0]

            model = Sequential()
            model.add(Dense(n_neurons, input_dim=proj_all_data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['hidden_activation']))
            model.add(Dense(proj_all_data.shape[1], init="uniform")) 
            model.add(Activation(trn_params.params['output_activation']))
        # end if layer == 2
        
        # Layer 3
        
        # end if layer == 3
        
        adam = Adam(lr=trn_params.params['learning_rate'], 
                    beta_1=trn_params.params['beta_1'],
                    beta_2=trn_params.params['beta_2'],
                    epsilon=trn_params.params['epsilon'])

        model.compile(loss='mean_squared_error', 
                      optimizer=adam,
                      metrics=['accuracy'])
        # Train model
        earlyStopping = callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=trn_params.params['patience'],
                                                verbose=trn_params.params['train_verbose'], 
                                                mode='auto')

        init_trn_desc = model.fit(norm_data[train_id], norm_trgt[train_id], 
                                  nb_epoch=trn_params.params['n_epochs'], 
                                  batch_size=trn_params.params['batch_size'],
                                  callbacks=[earlyStopping], 
                                  verbose=trn_params.params['verbose'],
                                  validation_data=(norm_data[test_id],
                                                   norm_trgt[test_id]),
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
    if not dev:        
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
    Function used to perform the training of a Classifier with one hidden layer.
    Topology(in terms of neurons): 
        InputDim(400) x n_neurons x OutputDim(Num of Classes)
'''
def NeuralTrainFunction(data=None, trgt=None, 
                        ifold=0, n_folds=2, n_neurons=10, 
                        trn_params=None, save_path='', dev=False):
    
    
    if n_neurons == 0:
        n_neurons = 1

    # turn targets in sparse mode
    trgt_sparse = np_utils.to_categorical(trgt.astype(int))
    
    # load or create cross validation ids
    CVO = trnparams.ClassificationFolds(folder=save_path,n_folds=n_folds,trgt=trgt,dev=dev)

    n_folds = len(CVO)
    n_inits = trn_params.params['n_inits']

    params_str = trn_params.get_params_str()
    
    analysis_str = 'NeuralNetwork'
    prefix_str = 'RawData'

    model_str = '%s/%s/%s_%i_folds_%s_%i_neurons'%(save_path,analysis_str,
                                                   prefix_str,n_folds,
                                                   params_str,n_neurons)

    
    if not dev:
        file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
        if os.path.exists(file_name):
            if trn_params.params['verbose']:
                print 'File %s exists'%(file_name)
            return 0
    else:
        file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
        if os.path.exists(file_name):
            if trn_params.params['verbose']:
                print 'File %s exists'%(file_name)
            return 0

    train_id, test_id = CVO[ifold]

    #normalize data based in train set
    if trn_params.params['norm'] == 'mapstd':
        scaler = preprocessing.StandardScaler().fit(data[train_id,:])
    elif trn_params.params['norm'] == 'mapstd_rob':
        scaler = preprocessing.RobustScaler().fit(data[train_id,:])
    elif trn_params.params['norm'] == 'mapminmax':
        scaler = preprocessing.MinMaxScaler().fit(data[train_id,:])

    norm_data = scaler.transform(data)

    best_init = 0
    best_loss = 999

    classifier = []
    trn_desc = {}

    for i_init in range(n_inits):
        print 'Neuron: %i - Fold %i of %i Folds -  Init %i of %i Inits'%(n_neurons, 
                                                                         ifold+1, 
                                                                         n_folds, 
                                                                         i_init+1,
                                                                         n_inits)
        model = Sequential()
        model.add(Dense(n_neurons, input_dim=data.shape[1], init="uniform"))
        model.add(Activation(trn_params.params['hidden_activation']))
        model.add(Dense(trgt_sparse.shape[1], init="uniform")) 
        model.add(Activation(trn_params.params['output_activation']))
        
        adam = Adam(lr=trn_params.params['learning_rate'], 
                    beta_1=trn_params.params['beta_1'],
                    beta_2=trn_params.params['beta_2'],
                    epsilon=trn_params.params['epsilon'])
        
        model.compile(loss='mean_squared_error', 
                      optimizer=adam,
                      metrics=['accuracy'])
        # Train model
        earlyStopping = callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=trn_params.params['patience'],
                                                verbose=trn_params.params['train_verbose'], 
                                                mode='auto')

        init_trn_desc = model.fit(norm_data[train_id], trgt_sparse[train_id], 
                                  nb_epoch=trn_params.params['n_epochs'], 
                                  batch_size=trn_params.params['batch_size'],
                                  callbacks=[earlyStopping], 
                                  verbose=trn_params.params['verbose'],
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
    if not dev:        
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
    Function used to perform the training of a Classifier and Novelty Detector
'''       
def NNNoveltyTrainFunction(data=None, trgt=None, inovelty=0, 
                           ifold=0, n_folds=2, n_neurons=1, 
                           trn_params=None, save_path='',
                           verbose=False, dev=False):
    
    # turn targets in sparse mode
    trgt_sparse = np_utils.to_categorical(trgt.astype(int))
    
    # load or create cross validation ids
    CVO = trnparams.NoveltyDetectionFolds(folder=save_path,n_folds=n_folds,trgt=trgt,dev=dev)
    
    if n_neurons == 0:
        n_neurons = 1

    n_folds = len(CVO[0])
    n_inits = trn_params.params['n_inits']
    model_prefix_str = 'RawData_%i_novelty'%(inovelty)
    analysis_path = 'NeuralNetwork'
    
    params_str = trn_params.get_params_str()
    
    model_str = '%s/%s/%s_%i_folds_%s_%i_neurons'%(save_path,analysis_path,
                                                   model_prefix_str,
                                                   n_folds,
                                                   params_str,
                                                   n_neurons)
    if not dev:
        file_name = '%s_%i_fold_model.h5'%(model_str,ifold)
    else:
        file_name = '%s_%i_fold_model_dev.h5'%(model_str,ifold)
        
    if trn_params.params['verbose']:
        print file_name
    
    # Check if the model has already been trained
    if not os.path.exists(file_name):
        if verbose:
            print 'Train Model'
        # training
        
        classifier = []
        trn_desc = {}
        
        train_id, test_id = CVO[inovelty][ifold]

        # normalize data based in train set
        if trn_params.params['norm'] == 'mapstd':
            scaler = preprocessing.StandardScaler().fit(data[train_id,:])
        elif trn_params.params['norm'] == 'mapstd_rob':
            scaler = preprocessing.RobustScaler().fit(data[train_id,:])
        elif trn_params.params['norm'] == 'mapminmax':
            scaler = preprocessing.MinMaxScaler().fit(data[train_id,:])

        norm_data = scaler.transform(data)
        
        best_init = 0
        best_loss = 999
        
        for i_init in range(n_inits):
            print 'Neuron: %i - Fold %i of %i Folds -  Init %i of %i Inits'%(n_neurons, 
                                                                             ifold+1, 
                                                                             n_folds, 
                                                                             i_init+1,
                                                                             n_inits)
            model = Sequential()
            model.add(Dense(n_neurons, input_dim=data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['hidden_activation']))
            model.add(Dense(trgt_sparse.shape[1], init="uniform")) 
            model.add(Activation(trn_params.params['output_activation']))
            
            adam = Adam(lr=trn_params.params['learning_rate'], 
                    beta_1=trn_params.params['beta_1'],
                    beta_2=trn_params.params['beta_2'],
                    epsilon=trn_params.params['epsilon'])
            
            model.compile(loss='mean_squared_error', 
                          optimizer=adam,
                          metrics=['accuracy'])
            
            # Train model
            earlyStopping = callbacks.EarlyStopping(monitor='val_loss', 
                                                    patience=trn_params.params['patience'],
                                                    verbose=trn_params.params['train_verbose'], 
                                                    mode='auto')

            init_trn_desc = model.fit(norm_data[train_id], trgt_sparse[train_id], 
                                      nb_epoch=trn_params.params['n_epochs'], 
                                      batch_size=trn_params.params['batch_size'],
                                      callbacks=[earlyStopping], 
                                      verbose=trn_params.params['verbose'],
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
        if not dev:        
            file_name = '%s_%i_fold_model.h5'%(model_str,ifold)
            classifier.save(file_name)
            file_name = '%s_%i_fold_trn_desc.jbl'%(model_str,ifold)
            joblib.dump([trn_desc],file_name,compress=9)
        else:
            file_name = '%s_%i_fold_model_dev.h5'%(model_str,ifold)
            classifier.save(file_name)
            file_name = '%s_%i_fold_trn_desc_dev.jbl'%(model_str,ifold)
            joblib.dump([trn_desc],file_name,compress=9)
    else:
        # The model has already been trained, so load the files
        if verbose: 
            print 'Load Model'
        classifier = Sequential()
        if not dev:
            file_name = '%s_%i_fold_model.h5'%(model_str,ifold)
        else:
            file_name = '%s_%i_fold_model_dev.h5'%(model_str,ifold)
        classifier = load_model(file_name)
        
        if not dev:
            file_name = '%s_%i_fold_trn_desc.jbl'%(model_str,ifold)
        else:
            file_name = '%s_%i_fold_trn_desc_dev.jbl'%(model_str,ifold)
        [trn_desc] = joblib.load(file_name)
        
    return [classifier,trn_desc]       


def SAENoveltyTrainFunction(data=None, trgt=None, inovelty=0, ifold=0, n_folds=2, n_neurons=1, layer = 1, hidden_neurons = [400],
                            trn_params=None, save_path='', verbose=False, dev=False):
    
    for i in range(len(hidden_neurons)):
        if hidden_neurons[i] == 0:
            hidden_neurons = 1
    
    if n_neurons == 0:
        n_neurons = 1

    # turn targets in sparse mode
    trgt_sparse = np_utils.to_categorical(trgt.astype(int))
    
    # load or create cross validation ids
    CVO = trnparams.NoveltyDetectionFolds(folder=save_path,n_folds=n_folds,trgt=trgt,dev=dev)
    
    n_folds = len(CVO[0])
    n_inits = trn_params.params['n_inits']
    prefix_str = 'RawData_%i_novelty'%(inovelty)
    analysis_str = 'StackedAutoEncoder'
    params_str = trn_params.get_params_str()
    
    if layer == 1:
        neurons_str = str(data.shape[1]) + 'x' + str(n_neurons)
        model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(save_path,analysis_str,
                                                          prefix_str,n_folds,
                                                          params_str,neurons_str)


        if not dev:
            file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                return 0
        else:
            file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                return 0
    if layer == 2:
        neurons_str = str(data.shape[1])
        for ineuron in hidden_neurons[:layer]:
            neurons_str = neurons_str + 'x' + str(ineuron)
        neurons_str = neurons_str + 'x' + str(n_neurons)
        model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(save_path,analysis_str,
                                                                prefix_str,n_folds,
                                                                params_str,neurons_str)


        if not dev:
            file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                return 0
        else:
            file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
            if os.path.exists(file_name):
                if trn_params.params['verbose']:
                    print 'File %s exists'%(file_name)
                return 0
            
    train_id, test_id = CVO[inovelty][ifold]

    #normalize data based in train set
    if trn_params.params['norm'] == 'mapstd':
        scaler = preprocessing.StandardScaler().fit(data[train_id,:])
    elif trn_params.params['norm'] == 'mapstd_rob':
        scaler = preprocessing.RobustScaler().fit(data[train_id,:])
    elif trn_params.params['norm'] == 'mapminmax':
        scaler = preprocessing.MinMaxScaler().fit(data[train_id,:])

    norm_data = scaler.transform(data)
    norm_trgt = norm_data
    
    best_init = 0
    best_loss = 999

    classifier = []
    trn_desc = {}
    
    for i_init in range(n_inits):
        print 'Neuron: %i - Fold %i of %i Folds -  Init %i of %i Inits'%(n_neurons, 
                                                                         ifold+1, 
                                                                         n_folds, 
                                                                         i_init+1,
                                                                         n_inits)
        if layer == 1:
            model = Sequential()
            model.add(Dense(n_neurons, input_dim=data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['hidden_activation']))
            model.add(Dense(data.shape[1], init="uniform")) 
            model.add(Activation(trn_params.params['output_activation']))
        if layer == 2:
            # Get the projection of the data from previous layer
            neurons_str = str(data.shape[1])
            for ineuron in hidden_neurons[:layer-1]:
                neurons_str = neurons_str + 'x' + str(ineuron)
            previous_model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(save_path,analysis_str,
                                                                    prefix_str,
                                                                    n_folds,
                                                                    params_str,
                                                                    neurons_str)
            if not dev:
                file_name = '%s_fold_%i_model.h5'%(previous_model_str,ifold)
            else:
                file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str,ifold)

            first_layer_model = load_model(file_name)

            get_layer_output = K.function([first_layer_model.layers[0].input],
                                          [first_layer_model.layers[1].output])
            proj_all_data = get_layer_output([norm_data])[0]

            model = Sequential()
            model.add(Dense(n_neurons, input_dim=proj_all_data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['hidden_activation']))
            model.add(Dense(proj_all_data.shape[1], init="uniform")) 
            model.add(Activation(trn_params.params['output_activation']))
        # end if layer == 2
        
        # Layer 3
        
        # end if layer == 3
        
        adam = Adam(lr=trn_params.params['learning_rate'], 
                    beta_1=trn_params.params['beta_1'],
                    beta_2=trn_params.params['beta_2'],
                    epsilon=trn_params.params['epsilon'])

        model.compile(loss='mean_squared_error', 
                      optimizer=adam,
                      metrics=['accuracy'])
        # Train model
        earlyStopping = callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=trn_params.params['patience'],
                                                verbose=trn_params.params['train_verbose'], 
                                                mode='auto')

        init_trn_desc = model.fit(norm_data[train_id], norm_trgt[train_id], 
                                  nb_epoch=trn_params.params['n_epochs'], 
                                  batch_size=trn_params.params['batch_size'],
                                  callbacks=[earlyStopping], 
                                  verbose=trn_params.params['verbose'],
                                  validation_data=(norm_data[test_id],
                                                   norm_trgt[test_id]),
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
    if not dev:        
        file_name = '%s_fold_%i_model.h5'%(model_str,ifold)
        classifier.save(file_name)
        file_name = '%s_fold_%i_trn_desc.jbl'%(model_str,ifold)
        joblib.dump([trn_desc],file_name,compress=9)
    else:
        file_name = '%s_fold_%i_model_dev.h5'%(model_str,ifold)
        classifier.save(file_name)
        file_name = '%s_fold_%i_trn_desc_dev.jbl'%(model_str,ifold)
        joblib.dump([trn_desc],file_name,compress=9)
    
    return [classifier, trn_desc]
    