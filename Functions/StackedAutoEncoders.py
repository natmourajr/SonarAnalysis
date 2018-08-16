# -*- coding: utf-8 -*-
"""
   Author: Vinícius dos Santos Mello viniciusdsmello at poli.ufrj.br
   Class created to implement a Stacked Autoencoder for Classification and Novelty Detection.
"""
import os
import numpy as np
import time

from sklearn.externals import joblib
from sklearn import preprocessing

from keras.models import Sequential
from keras import regularizers
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam, SGD
import keras.callbacks as callbacks
from keras.utils import np_utils
from keras.models import load_model
# from keras import backend as K
from keras import losses

from Functions.MetricsLosses import kullback_leibler_divergence
from Functions.lossWeights import getGradientWeights

import multiprocessing

import tensorflow as tf
# from keras.backend import tensorflow_backend as K

num_processes = multiprocessing.cpu_count()

# with tf.Session(config=tf.ConfigProto(
#                    intra_op_parallelism_threads=2)) as sess:
#    K.set_session(sess)


class StackedAutoEncoders:
    def __init__(self, parameters=None, save_path='', cvo=None, inovelty=0):
        self.save_path = save_path
        self.inovelty = inovelty
        self.parameters = parameters

        # Distinguish between a SAE for Novelty Detection and SAE for just Classification
        if bool(self.parameters["NoveltyDetection"]):
            self.cvo = cvo[self.inovelty]
            self.sae_prefix_str = "sae_model_%i_novelty" % self.inovelty
            self.classifier_prefix_str = "classifier_model_%i_novelty" % self.inovelty
        else:
            self.cvo = cvo
            self.sae_prefix_str = "sae_model"
            self.classifier_prefix_str = "classifier_model"

        # Choose optmizer algorithm
        if self.parameters["OptmizerAlgorithm"]["name"] == 'SGD':
            self.optmizer = SGD(lr=self.parameters["OptmizerAlgorithm"]["parameters"]["learning_rate"],
                                nesterov=self.parameters["OptmizerAlgorithm"]["parameters"]['nesterov'])

        elif self.parameters["OptmizerAlgorithm"]["name"] == 'Adam':
            self.optmizer = Adam(lr=self.parameters["OptmizerAlgorithm"]["parameters"]["learning_rate"],
                                 beta_1=self.parameters["OptmizerAlgorithm"]["parameters"]["beta_1"],
                                 beta_2=self.parameters["OptmizerAlgorithm"]["parameters"]["beta_2"],
                                 epsilon=self.parameters["OptmizerAlgorithm"]["parameters"]["epsilon"])
        else:
            self.optmizer = self.parameters["OptmizerAlgorithm"]["name"]

        # Choose loss functions
        if self.parameters["HyperParameters"]["loss"] == "kullback_leibler_divergence":
            self.lossFunction = kullback_leibler_divergence
        else:
            self.lossFunction = self.parameters["HyperParameters"]["loss"]
        losses.custom_loss = self.lossFunction


'''
        Method that creates a string in the format: (InputDimension)x(1º Layer Dimension)x...x(Nº Layer Dimension)
    '''


def get_neurons_str(self, data, hidden_neurons=None):
    if hidden_neurons is None:
        hidden_neurons = [1]
    neurons_str = str(data.shape[1])
    for ineuron in hidden_neurons:
        neurons_str = neurons_str + 'x' + str(ineuron)
    return neurons_str


'''
        Method that preprocess data normalizing it according to "norm" parameter.
    '''


def normalize_data(self, data, ifold):
    # normalize data based in train set
    train_id, test_id = self.cvo[ifold]
    if self.parameters["HyperParameters"]["norm"] == 'mapstd':
        scaler = preprocessing.StandardScaler().fit(data[train_id, :])
    elif self.parameters["HyperParameters"]["norm"] == 'mapstd_rob':
        scaler = preprocessing.RobustScaler().fit(data[train_id, :])
    elif self.parameters["HyperParameters"]["norm"] == 'mapminmax':
        scaler = preprocessing.MinMaxScaler().fit(data[train_id, :])
    else:
        return data
    norm_data = scaler.transform(data)

    return norm_data


'''
        Method that return the Stacked AutoEncoder model
    '''


def get_model(self, data, trgt, hidden_neurons=None, layer=1, ifold=0):
    if hidden_neurons is None:
        hidden_neurons = [1]
    if layer > len(hidden_neurons):
        print "[-] Error: The parameter layer must be less or equal to the size of list hidden_neurons"
        return 1
    if layer == 1:
        neurons_str = self.get_neurons_str(data, hidden_neurons[:layer])

        previous_model_str = os.path.join(self.save_path, self.sae_prefix_str + "_{}_neurons".format(neurons_str))

        file_name = '%s_fold_%i_model.h5' % (previous_model_str, ifold)

        # Check if previous layer model was trained
        if not os.path.exists(file_name):
            self.train_layer(data=data,
                            trgt=trgt,
                            ifold=ifold,
                            hidden_neurons=hidden_neurons[:layer],
                            layer=layer
                            )

        model = load_model(file_name,
                           custom_objects={'%s' % self.parameters["HyperParameters"]["loss"]: self.lossFunction})
    elif layer > 1:
        layer_models = {}
        layer_encoder_weights = {}
        layer_decoder_weights = {}

        for ilayer in range(1, layer + 1):
            neurons_str = self.get_neurons_str(data, hidden_neurons[:ilayer])
            previous_model_str = os.path.join(self.save_path,
                                              self.sae_prefix_str + "_{}_neurons".format(neurons_str))

            file_name = '%s_fold_%i_model.h5' % (previous_model_str, ifold)

            # Check if previous layer model was trained
            if not os.path.exists(file_name):
                self.train_layer(data=data,
                                trgt=trgt,
                                ifold=ifold,
                                hidden_neurons=hidden_neurons[:ilayer],
                                layer=ilayer
                                )

            layer_models[ilayer] = load_model(file_name, custom_objects={
                '%s' % self.parameters["HyperParameters"]["loss"]: self.lossFunction})

            layer_encoder_weights[ilayer] = layer_models[ilayer].layers[0].get_weights()
            layer_decoder_weights[ilayer] = layer_models[ilayer].layers[2].get_weights()

        model = Sequential()
        # Encoder
        for ilayer in range(1, layer + 1):
            if ilayer == 1:
                model.add(Dense(hidden_neurons[ilayer - 1], input_dim=data.shape[1],
                                weights=layer_encoder_weights[ilayer], trainable=False))
                model.add(Activation(self.parameters["HyperParameters"]["hidden_activation"]))
            else:
                model.add(Dense(hidden_neurons[ilayer - 1], weights=layer_encoder_weights[ilayer], trainable=False))
                model.add(Activation(self.parameters["HyperParameters"]["hidden_activation"]))
        # Decoder
        for ilayer in range(layer - 1, 0, -1):
            model.add(Dense(hidden_neurons[ilayer - 1], weights=layer_decoder_weights[ilayer + 1], trainable=False))
            model.add(Activation(self.parameters["HyperParameters"]["output_activation"]))

        model.add(Dense(data.shape[1], weights=layer_decoder_weights[ilayer], trainable=False))
        model.add(Activation(self.parameters["HyperParameters"]["output_activation"]))

    return model


'''
        Method that returns the encoder of an intermediate layer.
    '''


def get_encoder(self, data, trgt, hidden_neurons=None, layer=1, ifold=0):
    if hidden_neurons is None:
        hidden_neurons = [1]
    for i in range(len(hidden_neurons)):
        if hidden_neurons[i] == 0:
            hidden_neurons[i] = 1

    if (layer <= 0) or (layer > len(hidden_neurons)):
        print "[-] Error: The parameter layer must be greater than zero and less" \
              " or equal to the length of list hidden_neurons"
        return -1

    # Turn trgt to one-hot encoding
    trgt_sparse = np_utils.to_categorical(trgt.astype(int))

    neurons_str = self.get_neurons_str(data, hidden_neurons[:layer]) + 'x' + str(trgt_sparse.shape[1])

    train_id, test_id = self.cvo[ifold]

    norm_data = self.normalize_data(data, ifold)

    best_init = 0
    best_loss = 999

    # Start the model
    model = Sequential()
    # Add layers
    for ilayer in range(1, layer + 1):
        # Get the weights of ilayer

        neurons_str = self.get_neurons_str(data, hidden_neurons[:layer])

        previous_model_str = os.path.join(self.save_path,
                                          self.sae_prefix_str + "_{}_neurons".format(neurons_str))

        file_name = '%s_fold_%i_model_dev.h5' % (previous_model_str, ifold)

        # Check if the layer was trained
        if not os.path.exists(file_name):
            self.train_layer(data=data,
                            trgt=data,
                            ifold=ifold,
                            layer=ilayer,
                            hidden_neurons=hidden_neurons[:ilayer])

        layer_model = load_model(file_name, custom_objects={
            '%s' % self.parameters["HyperParameters"]["loss"]: self.lossFunction})
        layer_weights = layer_model.layers[0].get_weights()
        if ilayer == 1:
            model.add(Dense(units=hidden_neurons[0], input_dim=norm_data.shape[1], weights=layer_weights,
                            trainable=True))
        else:
            model.add(Dense(units=hidden_neurons[ilayer - 1], weights=layer_weights, trainable=True))

        model.add(Activation(self.parameters["HyperParameters"]["hidden_activation"]))

    return model


'''
        Method used to perform the layerwise algorithm to train the SAE
    '''


def train_layer(self, data=None, trgt=None, ifold=0, hidden_neurons=None, layer=1):
    # Change elements equal to zero to one
    if hidden_neurons is None:
        hidden_neurons = [400]
    for i in range(len(hidden_neurons)):
        if hidden_neurons[i] == 0:
            hidden_neurons[i] = 1
    if (layer <= 0) or (layer > len(hidden_neurons)):
        print "[-] Error: The parameter layer must be greater than zero and less " \
              "or equal to the length of list hidden_neurons"
        return -1

    if self.parameters['verbose']:
        print '[+] Using %s as optmizer algorithm' % self.parameters['optmizerAlgorithm']

    neurons_str = self.get_neurons_str(data, hidden_neurons[:layer])

    model_str = os.path.join(self.save_path, self.sae_prefix_str + "_{}_neurons".format(neurons_str))

    file_name = '%s_fold_%i_model.h5' % (model_str, ifold)
    if os.path.exists(file_name):
        if self.parameters['verbose']:
            print 'File %s exists' % file_name
        # load model
        file_name = '%s_fold_%i_model.h5' % (model_str, ifold)
        classifier = load_model(file_name, custom_objects={
            '%s' % self.parameters["HyperParameters"]["loss"]: self.lossFunction})
        file_name = '%s_fold_%i_trn_desc.jbl' % (model_str, ifold)
        trn_desc = joblib.load(file_name)
        return ifold, classifier, trn_desc

    train_id, test_id = self.cvo[ifold]

    norm_data = self.normalize_data(data, ifold)

    best_init = 0
    best_loss = 999

    classifier = []
    trn_desc = {}

    for i_init in range(self.n_inits):
        print 'Autoencoder - Layer: %i - Topology: %s - Fold %i of %i Folds -  Init %i of %i Inits' % (layer,
                                                                                                       neurons_str,
                                                                                                       ifold + 1,
                                                                                                       self.n_folds,
                                                                                                       i_init + 1,
                                                                                                       self.n_inits)
        model = Sequential()
        proj_all_data = norm_data
        if layer == 1:

            model.add(Dense(units=hidden_neurons[layer - 1], input_dim=data.shape[1],
                            kernel_initializer=self.parameters["HyperParameters"]["kernel_initializer"]))
            model.add(Activation(self.parameters["HyperParameters"]["hidden_activation"]))
            model.add(Dense(units=data.shape[1], input_dim=hidden_neurons[layer - 1],
                            kernel_initializer=self.parameters["HyperParameters"]["kernel_initializer"]))
            model.add(Activation(self.parameters["HyperParameters"]["output_activation"]))
        elif layer > 1:
            for ilayer in range(1, layer):
                neurons_str = self.get_neurons_str(data, hidden_neurons[:ilayer])

                previous_model_str = os.path.join(self.save_path,
                                                  self.sae_prefix_str + "_{}_neurons".format(neurons_str))

                file_name = '%s_fold_%i_model.h5' % (previous_model_str, ifold)

                # Check if previous layer model was trained
                if not os.path.exists(file_name):
                    self.train_layer(data=data,
                                    trgt=trgt,
                                    ifold=ifold,
                                    hidden_neurons=hidden_neurons[:ilayer],
                                    layer=ilayer)

                layer_model = load_model(file_name, custom_objects={
                    '%s' % self.parameters["HyperParameters"]["loss"]: self.lossFunction})

                get_layer_output = K.function([layer_model.layers[0].input],
                                              [layer_model.layers[1].output])
                # Projection of layer
                proj_all_data = get_layer_output([proj_all_data])[0]

            model.add(Dense(units=hidden_neurons[layer - 1], input_dim=proj_all_data.shape[1],
                            kernel_initializer=self.parameters["HyperParameters"]["kernel_initializer"]))
            model.add(Activation(self.parameters["HyperParameters"]["hidden_activation"]))

            if bool(self.parameters["HyperParameters"]["dropout"]):
                model.add(Dropout(int(self.parameters["HyperParameters"]["dropout_parameter"])))

                if self.parameters["HyperParameters"]["regularization"] == "l1":
                    model.add(Dense(units=proj_all_data.shape[1], input_dim=hidden_neurons[layer - 1],
                                    kernel_initializer=self.parameters["HyperParameters"]["kernel_initializer"],
                                    kernel_regularizer=regularizers.l1(
                                        self.parameters["HyperParameters"]["regularization_parameter"])))

                elif self.parameters["HyperParameters"]["regularization"] == "l2":
                    model.add(Dense(units=proj_all_data.shape[1], input_dim=hidden_neurons[layer - 1],
                                    kernel_initializer=self.parameters["HyperParameters"]["kernel_initializer"],
                                    kernel_regularizer=regularizers.l2(
                                        self.parameters["HyperParameters"]["regularization_parameter"])))

                else:
                    model.add(Dense(units=proj_all_data.shape[1], input_dim=hidden_neurons[layer - 1],
                                    kernel_initializer=self.parameters["HyperParameters"]["kernel_initializer"]))
                if bool(self.parameters["HyperParameters"]["dropout"]):
                    model.add(Dropout(int(self.parameters["HyperParameters"]["dropout_parameter"])))

                model.add(Activation(self.parameters["HyperParameters"]["output_activation"]))

                norm_data = proj_all_data
                # end of elif layer > 1:

                model.compile(loss=self.lossFunction,
                              optimizer=self.optmizer,
                              metrics=self.parameters["HyperParameters"]["metrics"])
                # Train model
                earlyStopping = callbacks.EarlyStopping(
                    monitor=self.parameters["callbacks"]["EarlyStopping"]["monitor"],
                    patience=self.parameters["callbacks"]["EarlyStopping"]["patience"],
                    verbose=self.verbose,
                    mode='auto')
                init_trn_desc = model.fit(norm_data[train_id], norm_data[train_id],
                                          epochs=self.parameters["HyperParameters"]["n_epochs"],
                                          batch_size=self.parameters["HyperParameters"]["batch_size"],
                                          callbacks=[earlyStopping],
                                          verbose=self.verbose,
                                          validation_data=(norm_data[test_id],
                                                           norm_data[test_id]),
                                          shuffle=True
                                          )
                if np.min(init_trn_desc.history['val_loss']) < best_loss:
                    best_init = i_init
                best_loss = np.min(init_trn_desc.history['val_loss'])
                classifier = model
                trn_desc['epochs'] = init_trn_desc.epoch

                for imetric in range(len(self.parameters["HyperParameters"]["metrics"])):
                    if self.parameters["HyperParameters"]["metrics"][imetric] == 'accuracy':
                        metric = 'acc'
                    else:
                        metric = self.parameters["HyperParameters"]["metrics"][imetric]
                trn_desc[metric] = init_trn_desc.history[metric]
                trn_desc['val_' + metric] = init_trn_desc.history['val_' + metric]

                trn_desc['loss'] = init_trn_desc.history['loss']
                trn_desc['val_loss'] = init_trn_desc.history['val_loss']

                # save model
                file_name = '%s_fold_%i_model.h5' % (model_str, ifold)
                classifier.save(file_name)
                file_name = '%s_fold_%i_trn_desc.jbl' % (model_str, ifold)
                joblib.dump([trn_desc], file_name, compress=9)

    return ifold, classifier, trn_desc


'''
        Method that return the classifier according to topology parsed
    '''


def load_classifier(self, data=None, trgt=None, hidden_neurons=None, layer=1, ifold=0, regularizer=None,
                   regularizer_param=None):
    if hidden_neurons is None:
        hidden_neurons = [1]
    for i in range(len(hidden_neurons)):
        if hidden_neurons[i] == 0:
            hidden_neurons[i] = 1
    if (layer <= 0) or (layer > len(hidden_neurons)):
        print "[-] Error: The parameter layer must be greater than zero and less " \
              "or equal to the length of list hidden_neurons"
        return -1

    # Turn trgt to one-hot encoding
    trgt_sparse = np_utils.to_categorical(trgt.astype(int))

    # load model
    neurons_str = self.get_neurons_str(data, hidden_neurons[:layer])

    model_str = os.path.join(self.save_path,
                             self.classifier_prefix_str + "_{}_neurons".format(self.inovelty, neurons_str))

    file_name = '%s_fold_%i_model.h5' % (model_str, ifold)
    try:
        classifier = load_model(file_name, custom_objects={
            '%s' % self.parameters["HyperParameters"]["loss"]: self.lossFunction})
    except ValueError:
        print '[-] Error: File or Directory not found. Path: {}'.format(file_name)
        return
    return classifier


'''
        Function used to do a Fine Tuning in Stacked Auto Encoder for Classification of the data
        hidden_neurons contains the number of neurons in the sequence: [FirstLayer, SecondLayer, ... ]
    '''


def train_classifier(self, data=None, trgt=None, ifold=0, hidden_neurons=None, layer=1, regularizer=None,
                    regularizer_param=None):
    if hidden_neurons is None:
        hidden_neurons = [1]
    for i in range(len(hidden_neurons)):
        if hidden_neurons[i] == 0:
            hidden_neurons[i] = 1

    if (layer <= 0) or (layer > len(hidden_neurons)):
        print "[-] Error: The parameter layer must be greater than zero and less " \
              "or equal to the length of list hidden_neurons"
        return -1

    # Turn trgt to one-hot encoding
    trgt_sparse = np_utils.to_categorical(trgt.astype(int))

    neurons_str = self.get_neurons_str(data, hidden_neurons[:layer])

    model_str = os.path.join(self.save_path,
                             self.classifier_prefix_str + "_{}_neurons".format(neurons_str))

    file_name = '{}_fold_{:d}_model.h5'.format(model_str, ifold)
    if os.path.exists(file_name):
        if self.parameters['verbose']:
            print 'File {} exists'.format(file_name)
        # load model
        file_name = '%s_fold_%i_model.h5' % (model_str, ifold)
        classifier = load_model(file_name, custom_objects={
            '%s' % self.parameters["HyperParameters"]["loss"]: self.lossFunction})
        file_name = '%s_fold_%i_trn_desc.jbl' % (model_str, ifold)
        trn_desc = joblib.load(file_name)
        return ifold, classifier, trn_desc

    train_id, test_id = self.cvo[ifold]

    norm_data = self.normalize_data(data, ifold)

    best_init = 0
    best_loss = 999

    classifier = []
    trn_desc = {}

    for i_init in range(self.n_inits):
        print 'Classifier - Layer: %i - Topology: %s - Fold: %i of %i Folds -  Init: %i of %i Inits' % (layer,
                                                                                                        neurons_str,
                                                                                                        ifold + 1,
                                                                                                        self.n_folds,
                                                                                                        i_init + 1,
                                                                                                        self.n_inits)
        # Start the model
        model = Sequential()
        # Add layers
        for ilayer in range(1, layer + 1):
            # Get the weights of ilayer
            neurons_str = self.get_neurons_str(data, hidden_neurons[:ilayer])

            previous_model_str = os.path.join(self.save_path,
                                              self.sae_prefix_str + "_{}_neurons".format(neurons_str))

            file_name = '%s_fold_%i_model_dev.h5' % (previous_model_str, ifold)

            # Check if the layer was trained
            if not os.path.exists(file_name):
                self.train_layer(data=data,
                                trgt=data,
                                ifold=ifold,
                                layer=ilayer,
                                hidden_neurons=hidden_neurons[:ilayer])

            layer_model = load_model(file_name, custom_objects={
                '%s' % self.parameters["HyperParameters"]["loss"]: self.lossFunction})
            layer_weights = layer_model.layers[0].get_weights()
            if ilayer == 1:
                model.add(Dense(units=hidden_neurons[0], input_dim=norm_data.shape[1], weights=layer_weights,
                                trainable=self.parameters["TechniqueParameters"]["allow_change_weights"]))
            else:
                model.add(Dense(units=hidden_neurons[ilayer - 1], weights=layer_weights,
                                trainable=self.parameters["TechniqueParameters"]["allow_change_weights"]))

            model.add(Activation(self.parameters["HyperParameters"]["hidden_activation"]))

        # Add Output Layer
        model.add(Dense(units=trgt_sparse.shape[1],
                        kernel_initializer=self.parameters["HyperParameters"]["kernel_initializer"]))
        model.add(Activation('softmax'))

        model.compile(loss=self.lossFunction,
                      optimizer=self.optmizer,
                      metrics=self.parameters["HyperParameters"]["metrics"])
        # Train model
        earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                                patience=self.parameters['patience'],
                                                verbose=self.parameters['train_verbose'],
                                                mode='auto')
        class_weights = getGradientWeights(trgt[train_id])
        init_trn_desc = model.fit(norm_data[train_id], trgt_sparse[train_id],
                                  epochs=self.parameters["HyperParameters"]["n_epochs"],
                                  batch_size=self.parameters["HyperParameters"]["batch_size"],
                                  callbacks=[earlyStopping],
                                  verbose=self.parameters['verbose'],
                                  validation_data=(norm_data[test_id], trgt_sparse[test_id]),
                                  shuffle=True,
                                  class_weight=class_weights
                                  )
        if np.min(init_trn_desc.history['val_loss']) < best_loss:
            best_init = i_init
            best_loss = np.min(init_trn_desc.history['val_loss'])
            classifier = model
            trn_desc['epochs'] = init_trn_desc.epoch

            for imetric in range(len(self.parameters["HyperParameters"]["metrics"])):
                if self.parameters["HyperParameters"]["metrics"][imetric] == 'accuracy':
                    metric = 'acc'
                else:
                    metric = self.parameters["HyperParameters"]["metrics"][imetric]
                trn_desc[metric] = init_trn_desc.history[metric]
                trn_desc['val_' + metric] = init_trn_desc.history['val_' + metric]

            trn_desc['loss'] = init_trn_desc.history['loss']
            trn_desc['val_loss'] = init_trn_desc.history['val_loss']

    # save model
    file_name = '%s_fold_%i_model.h5' % (model_str, ifold)
    classifier.save(file_name)
    file_name = '%s_fold_%i_trn_desc.jbl' % (model_str, ifold)
    joblib.dump([trn_desc], file_name, compress=9)

    return ifold, classifier, trn_desc
