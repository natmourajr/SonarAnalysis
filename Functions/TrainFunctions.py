"""
    Author: Vinicius dos Santos Mello
            viniciusdsmello@poli.ufrj.br

    Description: This file contains functions used for Neural Network training.
"""
import os
from warnings import warn
import pandas as pd
import numpy as np
import time

from sklearn.externals import joblib
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
import keras.callbacks as callbacks
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

import multiprocessing

import Functions.NpUtils.DataTransformation
from Functions import TrainParameters as trnparams, SystemIO, CrossValidation
from Functions.ConvolutionalNeuralNetworks import ConvolutionPaths

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

    def loadModels(self, model_list, model_initializer):
        self.models = [model_initializer(model) for model in model_list]

    def loadFolds(self, folds):
        self.fold_config = folds
        self.n_folds = len(folds)

    def train(self, transform_fn, preprocessing_fn, fold_mode='shuffleRuns', fold_balance='class_weights', novelty_classes= None,
              verbose=(0, 0, 0)):

        novelty_classes = novelty_classes or [None]
        data, trgt, class_labels = self.dataset

        n_models = len(self.models)
        for i_model, model in enumerate(self.models):
            if verbose[0]:
                print 'Model %i of %i' % (i_model + 1, n_models)
                trained_folds = []
                # model.pprint()

            status = model.selectFoldConfig(self.n_folds, fold_mode, fold_balance)
            if model.status is 'Trained':
                print "Already trained for current fold configuration"
                continue
            elif model.status is 'Recovery':
                print "Resuming Training"
                if novelty_classes is None:
                    pass
                    #trained_folds = [self._parseFoldNumber(fold_file) for fold_file in os.listdir(model.model_files)]
                else:
                    pass

            start = time.time()
            model_results = np.array(
                map(lambda cls: self._fitModel(model,
                                               data,
                                               trgt,
                                               class_labels,
                                               transform_fn,
                                               preprocessing_fn,
                                               novelty_cls=cls,
                                               verbose=verbose[1]),
                    novelty_classes),
                dtype=object)

            self._saveResults(model, model_results, novelty_classes, class_labels)

            if verbose[0]:
                end = time.time()
                print 'Training: %i (seg) / %i (min) /%i (hours)' % (
                end - start, (end - start) / 60, (end - start) / 3600)

    def _fitModel(self, model, data, trgt, class_labels, transform_fn, preprocessing_fn=None, novelty_cls=None,
                  verbose=False):
        n_classes = len(class_labels)

        if not novelty_cls is None:
            novelty_path_offset = '/' + class_labels[novelty_cls] + '/'
        else:
            novelty_path_offset = ''

        if model.status == 'Untrained':
            model_history = np.empty(self.n_folds, dtype=np.ndarray)
            model_predictions = np.empty(self.n_folds, dtype=np.ndarray)
        else:
            model_history = np.load(model.model_recovery_history)
            model_predictions = np.load(model.model_recovery_predictions)
        trained_folds = map(self._parseFoldNumber, model.trained_folds_files)

        print model.status
        print trained_folds

        for fold_count, (train_index, test_index) in enumerate(self.fold_config):
            if fold_count in trained_folds:
                continue

            x_train , y_train = transform_fn(all_data=data, all_trgt=trgt,
                                            index_info=train_index, info='train')
            x_test, y_test = transform_fn(all_data=data, all_trgt=trgt,
                                          index_info=test_index, info='val')

            if preprocessing_fn is not None:
                x_train, y_train = preprocessing_fn(x_train, y_train)
                x_test, y_test = preprocessing_fn(x_test, y_test)

            x_train = x_train[y_train != novelty_cls]
            x_test = x_test[y_test != novelty_cls]

            y_train = y_train[y_train != novelty_cls]
            y_test = y_test[y_test != novelty_cls]

            class_weights = self._getGradientWeights(y_train, mode='standard', novelty_cls=novelty_cls)

            y_test = Functions.NpUtils.DataTransformation.trgt2categorical(y_test, n_classes)
            y_train = Functions.NpUtils.DataTransformation.trgt2categorical(y_train, n_classes)

            nv_mask = np.ones(y_train.shape[1], dtype=bool)
            #nv_mask[novelty_cls or []] = False
            if not novelty_cls is None:
                nv_mask[novelty_cls] = False

            if verbose:
                print "Fold: " + str(fold_count) + '\n'
                print "Class Weights:"
                for cls in class_weights:
                    print "\t %s: %i%%" % (cls, 100 * class_weights[cls])

            # PASSAR PARA TRAINPARAMETERS
            # TODO pass novelty path creation to ModelPath class
            if not SystemIO.exists(model.model_best + novelty_path_offset):
                SystemIO.mkdir(model.model_best + novelty_path_offset)
            if not SystemIO.exists(model.model_files + novelty_path_offset):
                SystemIO.mkdir(model.model_files + novelty_path_offset)

            bestmodel = callbacks.ModelCheckpoint(model.model_best + novelty_path_offset + '/%i_fold.h5' % fold_count,
                                                  monitor='val_loss', mode='min', verbose=1, save_best_only=True)

            stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
            model.build()
            model_history[fold_count] = model.fit(x_train, y_train[:, nv_mask],
                                                  validation_data=(x_test, y_test[:, nv_mask]),
                                                  callbacks=[bestmodel, stopping],
                                                  class_weight=class_weights, verbose=verbose)

            #model_predictions[fold_count] = model.predict(x_test)

            model.save(model.model_files + novelty_path_offset + '/%i_fold.h5' % fold_count)
            model.model = load_model(model.model_best + novelty_path_offset + '/%i_fold.h5' % fold_count)

            model_predictions[fold_count] = model.predict(x_test)
            if not novelty_cls is None:
                model_predictions[fold_count] = np.concatenate(
                    [model_predictions[fold_count][:, :novelty_cls],
                     np.repeat(np.nan, model_predictions[fold_count].shape[0])[:, np.newaxis],
                     model_predictions[fold_count][:, novelty_cls:]],
                    axis=1)

            np.save(model.model_recovery_history, model_history)
            np.save(model.model_recovery_predictions, model_predictions)

            if K.backend() == 'tensorflow':  # solve tf memory leak
                K.clear_session()

        os.remove(model.model_recovery_predictions)
        os.remove(model.model_recovery_history)
        return model_history, model_predictions

    @staticmethod
    def _parseFoldNumber(fold_str):
        warn("Limit : [0,9]!")
        return int(fold_str[0])

    @staticmethod
    def _saveResults(model, model_results, novelty_classes, class_labels):
        history_ar = [history for history, _ in model_results]
        predictions_ar = [predictions for _, predictions in model_results]
        # print [[[nv_cls], [range(predictions.shape[0])]]
        #                   for nv_cls, prediction in zip(novelty_classes, predictions_ar)]
        predictions_pd = [[pd.DataFrame(fold_prediction, columns=class_labels.values(),
                                       index= pd.MultiIndex.from_product([['fold_%i' % i_fold], [nv_cls], range(fold_prediction.shape[0])]))
                          for i_fold, fold_prediction in enumerate(prediction)] for nv_cls, prediction in zip(novelty_classes, predictions_ar)]
        history_pd = [[pd.DataFrame(fold_history, index=pd.MultiIndex.from_product(
                                            [['fold_%i' % i_fold], [nv_cls], range(len(fold_history['loss']))]))
                           for i_fold, fold_history in enumerate(history)] for nv_cls, history in
                          zip(novelty_classes, history_ar)]

        pd_pred = pd.concat([pd.concat(predictions) for predictions in predictions_pd])
        pd_hist = pd.concat([pd.concat(predictions) for predictions in history_pd])

        hist_file = model.model_history + '/hist.csv'
        preds_file = model.model_predictions + '/pred.csv'

        pd_pred.to_csv(preds_file, sep=',')
        pd_hist.to_csv(hist_file, sep=',')

    def _getGradientWeights(self, y_train, mode = 'standard', novelty_cls=None):
        cls_indices, event_count = np.unique(np.array(y_train), return_counts=True)
        min_class = min(event_count)
        if not novelty_cls is None:
            cls_indices = [cls if cls < novelty_cls else cls - 1
                           for cls in cls_indices if cls != novelty_cls]
        print cls_indices
        return {cls_index: float(min_class) / cls_count
                for cls_index, cls_count in zip(cls_indices, event_count)}