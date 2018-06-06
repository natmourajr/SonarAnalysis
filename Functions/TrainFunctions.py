"""
    Author: Vinicius dos Santos Mello
            viniciusdsmello@poli.ufrj.br

    Description: This file contains functions used for Neural Network training.
"""

import os
import pickle
import warnings
from abc import abstractmethod

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
from keras import backend as K, Model

import multiprocessing

from Functions import TrainParameters as trnparams, SystemIO, DataHandler
from Functions.TrainParameters import TrnParamsConvolutional
from Functions.TrainPaths import ConvolutionPaths, ModelPaths

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
        # Add fourth hidden layer
        if(len(hidden_neurons) > 3):
             # Get the weights of fourth layer
            layer = 4
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

            fourth_layer_model = load_model(file_name)
            fourth_layer = fourth_layer_model.layers[0]
            fourth_layer_weights = fourth_layer.get_weights()

            model.add(Dense(hidden_neurons[3], weights=fourth_layer_weights, trainable=True))
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

    else:
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

            ################################
            # Check if previous model was trained
            ################################

            first_layer_model = load_model(file_name)

            get_layer_output = K.function([first_layer_model.layers[0].input],
                                          [first_layer_model.layers[1].output])
            proj_all_data = get_layer_output([norm_data])[0]

            model = Sequential()
            model.add(Dense(n_neurons, input_dim=proj_all_data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['hidden_activation']))
            model.add(Dense(proj_all_data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['output_activation']))
            norm_data = proj_all_data
        # end if layer == 2
        if layer == 3:
            # Load first layer model
            neurons_str = str(data.shape[1])
            for ineuron in hidden_neurons[:layer-2]:
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

            get_first_layer_output = K.function([first_layer_model.layers[0].input],
                                          [first_layer_model.layers[1].output])

            # Projection of first layer
            first_proj_all_data = get_first_layer_output([norm_data])[0]

            # Load second layer model
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

            second_layer_model = load_model(file_name)

            get_second_layer_output = K.function([second_layer_model.layers[0].input],
                                          [second_layer_model.layers[1].output])

            # Projection of second layer
            second_proj_all_data = get_second_layer_output([first_proj_all_data])[0]

            model = Sequential()
            model.add(Dense(n_neurons, input_dim=second_proj_all_data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['hidden_activation']))
            model.add(Dense(second_proj_all_data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['output_activation']))
            norm_data = second_proj_all_data
        # end if layer == 3
        if layer == 4:
            # Load first layer model
            neurons_str = str(data.shape[1])
            for ineuron in hidden_neurons[:layer-3]:
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

            get_first_layer_output = K.function([first_layer_model.layers[0].input],
                                          [first_layer_model.layers[1].output])

            # Projection of first layer
            first_proj_all_data = get_first_layer_output([norm_data])[0]

            # Load second layer model
            neurons_str = str(data.shape[1])
            for ineuron in hidden_neurons[:layer-2]:
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

            second_layer_model = load_model(file_name)

            get_second_layer_output = K.function([second_layer_model.layers[0].input],
                                          [second_layer_model.layers[1].output])

            # Projection of second layer
            second_proj_all_data = get_second_layer_output([first_proj_all_data])[0]

            # Load third layer model
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

            third_layer_model = load_model(file_name)

            get_third_layer_output = K.function([third_layer_model.layers[0].input],
                                          [third_layer_model.layers[1].output])

            # Projection of third layer
            third_proj_all_data = get_third_layer_output([second_proj_all_data])[0]

            model = Sequential()
            model.add(Dense(n_neurons, input_dim=third_proj_all_data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['hidden_activation']))
            model.add(Dense(third_proj_all_data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['output_activation']))
            norm_data = third_proj_all_data
        #end if layer == 4

        # Optimizer
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

        init_trn_desc = model.fit(norm_data[train_id], norm_data[train_id],
                                  nb_epoch=trn_params.params['n_epochs'],
                                  batch_size=trn_params.params['batch_size'],
                                  callbacks=[earlyStopping],
                                  verbose=trn_params.params['verbose'],
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
    else:
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
        # end if layer == 1

        if layer == 2:
            # Load second layer model
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
            # Projection of first layer
            proj_all_data = get_layer_output([norm_data])[0]

            model = Sequential()
            model.add(Dense(n_neurons, input_dim=proj_all_data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['hidden_activation']))
            model.add(Dense(proj_all_data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['output_activation']))

            norm_data = proj_all_data
        # end if layer == 2

        if layer == 3:
            # Load first layer model
            neurons_str = str(data.shape[1])
            for ineuron in hidden_neurons[:layer-2]:
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

            get_first_layer_output = K.function([first_layer_model.layers[0].input],
                                          [first_layer_model.layers[1].output])

            # Projection of first layer
            first_proj_all_data = get_first_layer_output([norm_data])[0]
            # Load second layer model
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

            second_layer_model = load_model(file_name)

            get_second_layer_output = K.function([second_layer_model.layers[0].input],
                                          [second_layer_model.layers[1].output])

            # Projection of second layer
            second_proj_all_data = get_second_layer_output([first_proj_all_data])[0]

            model = Sequential()
            model.add(Dense(n_neurons, input_dim=second_proj_all_data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['hidden_activation']))
            model.add(Dense(second_proj_all_data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['output_activation']))

            norm_data = second_proj_all_data
        # end if layer == 3
        if layer == 4:
            # Load first layer model
            neurons_str = str(data.shape[1])
            for ineuron in hidden_neurons[:layer-3]:
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

            get_first_layer_output = K.function([first_layer_model.layers[0].input],
                                          [first_layer_model.layers[1].output])

            # Projection of first layer
            first_proj_all_data = get_first_layer_output([norm_data])[0]

            # Load second layer model
            neurons_str = str(data.shape[1])
            for ineuron in hidden_neurons[:layer-2]:
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

            second_layer_model = load_model(file_name)

            get_second_layer_output = K.function([second_layer_model.layers[0].input],
                                          [second_layer_model.layers[1].output])

            # Projection of second layer
            second_proj_all_data = get_second_layer_output([first_proj_all_data])[0]

            # Load third layer model
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

            third_layer_model = load_model(file_name)

            get_third_layer_output = K.function([third_layer_model.layers[0].input],
                                          [third_layer_model.layers[1].output])

            # Projection of third layer
            third_proj_all_data = get_third_layer_output([second_proj_all_data])[0]

            model = Sequential()
            model.add(Dense(n_neurons, input_dim=third_proj_all_data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['hidden_activation']))
            model.add(Dense(third_proj_all_data.shape[1], init="uniform"))
            model.add(Activation(trn_params.params['output_activation']))

            norm_data = third_proj_all_data
        #end if layer == 4

        # Optmizer
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

        init_trn_desc = model.fit(norm_data[train_id], norm_data[train_id],
                                  nb_epoch=trn_params.params['n_epochs'],
                                  batch_size=trn_params.params['batch_size'],
                                  callbacks=[earlyStopping],
                                  verbose=trn_params.params['verbose'],
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

def SAEClassifierNoveltyTrainFunction(data=None, trgt=None, inovelty=0, ifold=0, n_folds=2, hidden_neurons = [400],
                                      trn_params=None, save_path='', verbose=False, dev=False):

    # turn targets in sparse mode
    trgt_sparse = np_utils.to_categorical(trgt.astype(int))

    # load or create cross validation ids
    CVO = trnparams.NoveltyDetectionFolds(folder=save_path,n_folds=n_folds,trgt=trgt,dev=dev)

    n_folds = len(CVO[inovelty])
    n_inits = trn_params.params['n_inits']
    prefix_str = 'RawData_%i_novelty'%(inovelty)
    analysis_str = 'StackedAutoEncoder'
    params_str = trn_params.get_params_str()

    neurons_str = str(data.shape[1])
    for ineuron in hidden_neurons:
        neurons_str = neurons_str + 'x' + str(ineuron)

    model_str = '%s/%s/%s_%i_folds_%s_%s_neurons'%(save_path,analysis_str,
                                                   prefix_str,n_folds,
                                                   params_str,neurons_str)

    if not dev:
        file_name = '%s_classifier_fold_%i_model.h5'%(model_str,ifold)
    else:
        file_name = '%s_classifier_fold_%i_model_dev.h5'%(model_str,ifold)

    if not os.path.exists(file_name):
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
            print 'Topology: %sx%i - Fold %i of %i Folds -  Init %i of %i Inits'%(neurons_str,
                                                                                  trgt_sparse.shape[1],
                                                                                  ifold+1,
                                                                                  n_folds,
                                                                                  i_init+1,
                                                                                  n_inits)


            # First Layer
            previous_model_str = '%s/%s/%s_%i_folds_%s_400x%i_neurons'%(save_path,analysis_str,
                                                                        prefix_str,
                                                                        n_folds,
                                                                        params_str,
                                                                        hidden_neurons[0])

            if not dev:
                file_name = '%s_fold_%i_model.h5'%(previous_model_str, ifold)
            else:
                file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str, ifold)

            # Get first layer weights
            first_layer_model = load_model(file_name)
            encoder_first_layer = first_layer_model.layers[0].get_weights()

            model = Sequential()

            # Encoder of first layer
            model.add(Dense(hidden_neurons[0], input_dim=norm_data.shape[1], weights=encoder_first_layer, trainable=False))
            model.add(Activation(trn_params.params['hidden_activation']))

            if len(hidden_neurons) > 1:
                # Second Layer
                previous_model_str = '%s/%s/%s_%i_folds_%s_400x%ix%i_neurons'%(save_path,analysis_str,
                                                                        prefix_str,
                                                                        n_folds,
                                                                        params_str,
                                                                        hidden_neurons[0],
                                                                        hidden_neurons[1])

                if not dev:
                    file_name = '%s_fold_%i_model.h5'%(previous_model_str, ifold)
                else:
                    file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str, ifold)
                # Get the second layer projection of data
                second_layer_model = load_model(file_name)
                encoder_second_layer = second_layer_model.layers[0].get_weights()

                # Encoder of second layer
                model.add(Dense(hidden_neurons[1], weights=encoder_second_layer, trainable=False))
                model.add(Activation(trn_params.params['hidden_activation']))

            if len(hidden_neurons) > 2:
                # Third Layer
                previous_model_str = '%s/%s/%s_%i_folds_%s_400x%ix%ix%i_neurons'%(save_path,analysis_str,
                                                                        prefix_str,
                                                                        n_folds,
                                                                        params_str,
                                                                        hidden_neurons[0],
                                                                        hidden_neurons[1],
                                                                        hidden_neurons[2])

                if not dev:
                    file_name = '%s_fold_%i_model.h5'%(previous_model_str, ifold)
                else:
                    file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str, ifold)

                # Get the third layer projection of data
                third_layer_model = load_model(file_name)
                encoder_third_layer = third_layer_model.layers[0].get_weights()

                # Encoder of third layer
                model.add(Dense(hidden_neurons[2], weights=encoder_third_layer, trainable=False))
                model.add(Activation(trn_params.params['hidden_activation']))

            if len(hidden_neurons) > 3:
                # Fourth Layer
                previous_model_str = '%s/%s/%s_%i_folds_%s_400x%ix%ix%ix%i_neurons'%(save_path,analysis_str,
                                                                        prefix_str,
                                                                        n_folds,
                                                                        params_str,
                                                                        hidden_neurons[0],
                                                                        hidden_neurons[1],
                                                                        hidden_neurons[2],
                                                                        hidden_neurons[3])

                if not dev:
                    file_name = '%s_fold_%i_model.h5'%(previous_model_str, ifold)
                else:
                    file_name = '%s_fold_%i_model_dev.h5'%(previous_model_str, ifold)

                # Get the third layer projection of data
                fourth_layer_model = load_model(file_name)
                encoder_fourth_layer = fourth_layer_model.layers[0].get_weights()

                # Encoder of fourth layer
                model.add(Dense(hidden_neurons[3], weights=encoder_fourth_layer, trainable=False))
                model.add(Activation(trn_params.params['hidden_activation']))

            model.add(Dense(trgt_sparse.shape[1], init="uniform"))
            model.add(Activation('softmax'))

            # Optmizer
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
            file_name = '%s_classifier_fold_%i_model.h5'%(model_str,ifold)
            classifier.save(file_name)
            file_name = '%s_classifier_fold_%i_trn_desc.jbl'%(model_str,ifold)
            joblib.dump([trn_desc],file_name,compress=9)
        else:
            file_name = '%s_classifier_fold_%i_model_dev.h5'%(model_str,ifold)
            classifier.save(file_name)
            file_name = '%s_classifier_fold_%i_trn_desc_dev.jbl'%(model_str,ifold)
            joblib.dump([trn_desc],file_name,compress=9)
    else:
        # The model has already been trained, so load the files
        if verbose:
            print 'Load Model'
        classifier = Sequential()
        if not dev:
            file_name = '%s_classifier_fold_%i_model.h5'%(model_str,ifold)
        else:
            file_name = '%s_classifier_fold_%i_model_dev.h5'%(model_str,ifold)
        classifier = load_model(file_name)

        if not dev:
            file_name = '%s_classifier_fold_%i_trn_desc.jbl'%(model_str,ifold)
        else:
            file_name = '%s_classifier_fold_%i_trn_desc_dev.jbl'%(model_str,ifold)
        [trn_desc] = joblib.load(file_name)

    return [classifier,trn_desc]


class ConvolutionTrainFunction(ConvolutionPaths):
    def __init__(self):
        # Path variables
        super(ConvolutionTrainFunction, self).__init__()

    def loadData(self, dataset):
        """Load dataset for model use

        Args:
        dataset(tuple): (samples, labels, unique(labels))
        """
        self.dataset = dataset

    def loadModels(self, model_list):
        self.models = model_list

    def loadFolds(self, n_folds):
        file_name = '%s/%i_folds.jbl' % (self.results_path, n_folds)
        print "Loading fold configuration"
        self.fold_config = SystemIO.load(file_name)
        self.n_folds = n_folds

    def train(self, verbose=(0, 0, 0)):
        # verify data structure
        data, trgt, class_labels = self.dataset
        # Generalize number of classes
        n_classes = len(class_labels)

        categorical_trgt = DataHandler.trgt2categorical(trgt, n_classes)

        #             if self.scale:
        #                 scaler function

        # implement failsafe files
        n_models = len(self.models)
        for i_model, model in enumerate(self.models):
            if verbose[0]:
                print 'Model %i of %i' % (i_model, n_models)
                model.pprint()

            status = model.selectFoldConfig(self.n_folds)
            if status is 'Trained':
                print "Already trained for current fold configuration"
                continue
            elif status is 'Recovery':
                print "Resuming Training"
                # trained_folds = #load recovery file
                raise NotImplementedError

            start = time.time()
            for fold_count, train_index, test_index in enumerate(self.fold_config):
                # if fold_count in trained_folds:
                #    continue

                x_train, y_train = data[train_index], categorical_trgt[train_index]
                x_test, y_test = data[test_index], categorical_trgt[test_index]

                if verbose[1]:
                    print "Fold: " + str(fold_count) + '\n'

                model.build(self.n_folds)
                fold_history = self.model.fit(x_train, y_train, x_test, y_test, verbose[2])

                # save model recovery
                model.save(model.recovery_file)
                # save model rec folds

                fold_predictions = model.predict(data)
                print fold_predictions
                print fold_history
                # SystemIO.save(fold_predictions, model.model_predictions + '/' + '%i_fold.csv' % fold_count)

                if not 'ModelCheckpoint' in model.callbacks:
                    warnings.warn('ModelCheckpoint Callback not found for current model.')
                    model.save(model.model_file)

            # delete model recovery
            # delete model fold recovery

            if verbose[0]:
                end = time.time()
                print 'Training: %i (seg) / %i (min) /%i (hours)' % (end - start, (end - start) / 60, (end - start) / 3600)


class _CNNModel(ModelPaths):
    def __init__(self, trnParams):
        if not isinstance(trnParams, TrnParamsConvolutional):
            raise ValueError('The parameters inserted must be an instance of TrnPararamsCNN'
                             '%s of type %s was passed.'
                             % (trnParams, type(trnParams)))

        self.model = None
        self.params = None

        self.path = ModelPaths(trnParams)
        if SystemIO.exists(self.path.model_path):
            trnParams = self.loadInfo()  # see __setstate__ and arb obj storage with joblib
            self.mountParams(trnParams)
        else:
            self.createFolders(self.results_path + '/' + trnParams.getParamPath())
            self.mountParams(trnParams)
            self.saveParams(trnParams)

    def summary(self):
        raise NotImplementedError

    def mountParams(self, trnParams):
        for param, value in trnParams:
            if param is None:
                raise ValueError('The parameters configuration received by _CNNModel must be complete'
                                 '%s was passed as NoneType.'
                                 % (param))
            setattr(self, param, value)

        # self.model_name = trnParams.getParamStr()

    def saveParams(self, trnParams):
        SystemIO.save(trnParams.toNpArray(), self.path.model_info_file)

    def loadInfo(self):
        params_array = SystemIO.load(self.path.model_info_file)
        return TrnParamsConvolutional(*params_array)

    @abstractmethod
    def load(self, file_path):
        pass

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def fit(self, x_train, y_train, validation_data=None, class_weight=None, verbose=0):
        pass

    @abstractmethod
    def evaluate(self, x_test, y_test, verbose=0):
        pass

    @abstractmethod
    def predict(self, data, verbose=0):
        pass


class KerasModel(_CNNModel):
    def __init__(self, trnParams):
        super(KerasModel, self).__init__(trnParams)

    def build(self):
        if self.model is not None:
            warnings.warn('Model is not empty and was already trained.\n'
                          'Run purge method for deleting the model variable',
                          Warning)

        self.purge()
        self.model = Sequential()
        for layer in self.layers:
            self.model.add(layer.toKerasFn())

        self.model.compile(optimizer=self.optimizer.toKerasFn(),
                           loss=self.loss,
                           metrics=self.metrics
                           )

    def fit(self, x_train, y_train, validation_data=None, class_weight=None, verbose=0):
        if self.model is None:
            raise StandardError('Model is not built. Run build method or load model before fitting')

        history = self.model.fit(x_train,
                                 y_train,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 callbacks=self.callbacks.toKerasFn(),
                                 validation_data=validation_data,
                                 class_weight=class_weight,
                                 verbose=verbose
                                 )
        self.history = history
        return history

    def evaluate(self, x_test, y_test, verbose=0):
        if self.model is None:
            raise StandardError('Model is not built. Run build method or load model before fitting')

        val_history = self.model.evaluate(x_test,
                                          y_test,
                                          batch_size=self.batch_size,
                                          verbose=verbose)
        self.val_history = val_history
        return val_history

    def get_layer_n_output(self, n, data):
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.layers[n - 1].output)
        intermediate_output = intermediate_layer_model.predict(data)
        return intermediate_output

    def predict(self, data, verbose=0):
        if self.model is None:
            raise StandardError('Model is not built. Run build method or load model before fitting')
        return self.model.predict(data, 1, verbose)  # ,steps)

    def load(self, file_path):
        self.model = load_model(file_path)

    def save(self, file_path):
        self.model.save(file_path)

    def purge(self):
        del self.model