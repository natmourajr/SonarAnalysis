'''
   Author: Pedro Henrique Braga Lisboa pedrolisboa at poli.ufrj.br
   This module gather utilities to implement a Convolutional Neural Network
   for Classification and Novelty Detection.
'''
import inspect
import os
import re
import time
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from abc import abstractmethod
from itertools import product, cycle
from warnings import warn
import seaborn as sns

import keras
import numpy as np
from keras import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection._search import BaseSearchCV, ParameterGrid, _check_param_grid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Functions import SystemIO, TrainParameters
from Functions.NpUtils.Scores import spIndex, recall_score
from Functions.SystemIO import load, exists, mkdir
from Functions.TrainParameters import TrnParamsConvolutional, CNNParams, Layers, Callbacks

from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.base import MetaEstimatorMixin

from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.exceptions import NotFittedError
from sklearn.utils import Parallel, delayed
from sklearn.externals import six
from sklearn.utils import check_random_state
from sklearn.utils.fixes import sp_version
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.fixes import _Mapping as Mapping, _Sequence as Sequence
from sklearn.utils.fixes import _Iterable as Iterable
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import indexable, check_is_fitted
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.deprecation import DeprecationDict
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.metrics.scorer import check_scoring


class _Path(object):
    MODEL_INFO_FILE_NAME = 'model_info.jbl'
    MODEL_STATE_FILE_NAME = 'model_state.h5'

    def __init__(self):
        pass

    def add(self, subfolder):
        # self.subfolders[subfolder] = self.home + '/' + sub-folder
        pass

    def createFolder(self):
        """Generate folders accordingly with the object structure"""
        raise NotImplementedError


class ConvolutionPaths(_Path):
    """Class with references to all sub-folders(models and results)
    required for the convolution models training and analysis"""
    results_path = os.getenv('PACKAGE_NAME')

    def __init__(self):
        super(ConvolutionPaths, self).__init__()
        self.model_paths = self.getModelPaths(self.results_path)

    def getModelPaths(self, init_path):
        try:
           path_list = os.listdir(init_path)
        except OSError:
            return

        if self.MODEL_INFO_FILE_NAME in path_list:
            return init_path
        return np.array([self.getModelPaths(init_path + '/' + path) for path in path_list]).flatten()

    def iterModelPaths(self, paramsMask):
        """Generates model paths attending the desired conditions"""

        if not isinstance(paramsMask, TrnParamsConvolutional):
            raise ValueError('The parameters inserted must be an instance of TrnParamsConvolutional'
                             '%s of type %s was passed.'
                             % (paramsMask, type(paramsMask)))

        for model_path in self.models_paths:
            model_params = load(model_path)

            if model_params.fitMask(paramsMask):
                yield model_path


class ModelPaths(ConvolutionPaths):
    """Class with references to all sub-folders(models and results)
        required for the models training """
    def __init__(self, trnParams,  results_path=None):
        super(ModelPaths, self).__init__()
        if not (isinstance(trnParams, TrnParamsConvolutional) or isinstance(trnParams, CNNParams)):
            raise ValueError('The parameters inserted must be an instance of TrnPararamsConvolutional'
                             '%s of type %s was passed.'
                             % (trnParams, type(trnParams)))
        self.results_path = results_path
        self.model_path = self.results_path + '/' + self.genPathStr(trnParams)
        # self.model_date = "%s_%s_%s_%s"%(now.year,now.month,now.day,now.hour)

        self.model_info_file = self.model_path + '/' + 'model_info.jbl'
        self.model_info_file_txt = self.model_path + '/' + 'model_info.txt'
        self.model_files = None
        self.model_best = None
        self.model_history = None
        self.model_predictions = None
        self.model_recovery_state = None
        self.model_recovery_folds = None

    def genPathStr(self, trnParams):
        # TODO implement definitive path

        return trnParams.getParamPath()

    def selectFoldConfig(self, n_folds, mode='shuffleRuns', balance='class_weights'):
        warn('Implement error handling')

        self.fold_path = self.model_path + '/' + '%s' % mode
        self.balance_mode = balance
        self.model_files = self.fold_path + '/' + 'states'
        self.model_best = self.fold_path + '/' + 'best_states'
        self.model_history = self.fold_path + '/' + 'history.csv'
        self.model_recovery_history = self.fold_path + '/' + '~history.npy'

        self.model_predictions = self.fold_path + '/' + 'predictions.csv'
        self.model_recovery_predictions = self.fold_path + '/' + '~predictions.npy'

        self.model_recovery_state = self.fold_path + '/' + '~rec_state.h5'
        self.model_recovery_folds = self.fold_path + '/' + '~rec_folds.jbl'


        if exists(self.model_files):
            length_tr = len(os.listdir(self.model_files))
            if length_tr < n_folds and length_tr > 0:
                self.status = 'Recovery'

            elif length_tr == n_folds:
                self.status = 'Trained'
            else:
                self.status = 'Untrained'
        # elif exists(self.model_recovery_state):
        #   return 'Recovery'
        else:
            self.createFolders(self.model_files, self.model_best)
            self.status = 'Untrained'

        self.trained_folds_files = os.listdir(self.model_files)

    def createFolders(self, *args):
        """Creates model folder structure from hyperparameters information"""
        for folder in args:
            mkdir(folder)

    def purgeFolder(self, subfolder=''):
        raise NotImplementedError


class _CNNModel(ModelPaths):
    """Base class for all Convolutional Neural Networks models

        This class serves as an interface between the parameter class and API-specific
        model classes (see KerasModel class).

       Implementations must define load, save, build, fit, evaluate and predict in order to interface with
       the desired API
    """

    def __init__(self, trnParams):
        super(_CNNModel, self).__init__(trnParams)
        if not (isinstance(trnParams, TrnParamsConvolutional) or isinstance(trnParams, CNNParams)):
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
        """Returns a summary of the network topology and training parameters"""
        raise NotImplementedError

    def mountParams(self, trnParams):
        """Parses the parameters to attributes of the instance

           trnParams (TrnParamsConvolutional): parameter object
        """
        for param, value in trnParams:
            if param is None:
                raise ValueError('The parameters configuration received by _CNNModel must be complete'
                                 '%s was passed as NoneType.'
                                 % (param))
            setattr(self, param, value)

        # self.model_name = trnParams.getParamStr()

    def saveParams(self, trnParams):
        """Save parameters into a pickle file"""

        SystemIO.save(trnParams.toNpArray(), self.path.model_info_file)

    def loadInfo(self):
        """Load parameters from a pickle file"""
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
    """Keras Sequential Model wrapper class

       Inherits from _CNNModel and adds details
       for interfacing with the Keras API
    """

    def __init__(self, trnParams):
        super(OldKerasModel, self).__init__(trnParams)

    def __call__(self, *args, **kwargs):
        self.__init__(*args)

    def build(self):
        """Compile Keras model using the parameters loaded into the instance"""

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

    def fit(self, x_train, y_train, callbacks, validation_data=None, class_weight=None,
            verbose=0, max_restarts=0, restart_tol=None, restart_monitor='spIndex'):
        """Model training routine

           x_train (numpy.nparray): model input data
           y_train (numpy.nparray): input data labels
           callbacks: manual callbacks insertion. It will be removed on future commits
           validation_data (<numpy.nparray, numpy.nparray>): tuple containing input and trgt
                                                             data for model validation. <input, trgt>
           class_weight (dict): dictionary mapping class indices to a floating point value,
                                used for weighting the loss function during training
                                (see ConvolutionTrainFunction.getClassWeights)

           max_restarts (int): max number of restarts if the desired conditions are not reached
           restart_tol: tolerance value for the restart condition
           restart_monitor: condition for restarting training

            :returns : dictionary containing loss and metrics for each epoch
        """

        # TODO pass restart to a Keras callback
        # TODO fix method signature removing manual callback selection
        if self.model is None:
            raise StandardError('Model is not built. Run build method or load model before fitting')

        while max_restarts >= 0:
            history = self.model.fit(x_train,
                                     y_train,
                                     epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     #callbacks=self.callbacks,
                                     callbacks=callbacks,
                                     validation_data=validation_data,
                                     class_weight=class_weight,
                                     verbose=verbose
                                     )

            if np.array(history.history[restart_monitor]).max() < restart_tol:
                print "Max: %f" % np.array(history.history[restart_monitor]).max()
                print "Restarting"
                self.build()  # reset model
                max_restarts -= 1
            else:
                break

        self.history = history.history
        return history.history

    def evaluate(self, x_test, y_test, verbose=0):
        """Model evaluation on a test set

            x_test: input data
            y_test: data labels

            :returns : evaluation results
        """

        if self.model is None:
            raise StandardError('Model is not built. Run build method or load model before fitting')

        test_results = self.model.evaluate(x_test,
                                          y_test,
                                          batch_size=self.batch_size,
                                          verbose=verbose)
        self.val_history = test_results
        return test_results

    def get_layer_n_output(self, n, data):
        """Returns the output of layer n of the model for the given data"""

        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.layers[n].output)
        intermediate_output = intermediate_layer_model.predict(data)
        return intermediate_output

    def predict(self, data, verbose=0):
        """Model evaluation on a data set

            :returns : model predictions (numpy.nparray / shape: <n_samples, n_outputs>
        """

        if self.model is None:
            raise StandardError('Model is not built. Run build method or load model before fitting')
        return self.model.predict(data, 1, verbose)  # ,steps)

    def load(self, file_path):
        """Load model info and weights from a .h5 file"""
        self.model = load_model(file_path)

    def save(self, file_path):
        """Save current model state and weights on an .h5 file"""
        self.model.save(file_path)

    def purge(self):
        """Delete model state"""
        del self.model

class OldKerasModel(_CNNModel):
    """Keras Sequential Model wrapper class

       Inherits from _CNNModel and adds details
       for interfacing with the Keras API
    """

    def __init__(self, trnParams):
        super(OldKerasModel, self).__init__(trnParams)

    def __call__(self, *args, **kwargs):
        self.__init__(*args)

    def build(self):
        """Compile Keras model using the parameters loaded into the instance"""

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

    def fit(self, x_train, y_train, callbacks, validation_data=None, class_weight=None,
            verbose=0, max_restarts=0, restart_tol=None, restart_monitor='spIndex'):
        """Model training routine

           x_train (numpy.nparray): model input data
           y_train (numpy.nparray): input data labels
           callbacks: manual callbacks insertion. It will be removed on future commits
           validation_data (<numpy.nparray, numpy.nparray>): tuple containing input and trgt
                                                             data for model validation. <input, trgt>
           class_weight (dict): dictionary mapping class indices to a floating point value,
                                used for weighting the loss function during training
                                (see ConvolutionTrainFunction.getClassWeights)

           max_restarts (int): max number of restarts if the desired conditions are not reached
           restart_tol: tolerance value for the restart condition
           restart_monitor: condition for restarting training

            :returns : dictionary containing loss and metrics for each epoch
        """

        # TODO pass restart to a Keras callback
        # TODO fix method signature removing manual callback selection
        if self.model is None:
            raise StandardError('Model is not built. Run build method or load model before fitting')

        while max_restarts >= 0:
            history = self.model.fit(x_train,
                                     y_train,
                                     epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     #callbacks=self.callbacks,
                                     callbacks=callbacks,
                                     validation_data=validation_data,
                                     class_weight=class_weight,
                                     verbose=verbose
                                     )

            if np.array(history.history[restart_monitor]).max() < restart_tol:
                print "Max: %f" % np.array(history.history[restart_monitor]).max()
                print "Restarting"
                self.build()  # reset model
                max_restarts -= 1
            else:
                break

        self.history = history.history
        return history.history

    def evaluate(self, x_test, y_test, verbose=0):
        """Model evaluation on a test set

            x_test: input data
            y_test: data labels

            :returns : evaluation results
        """

        if self.model is None:
            raise StandardError('Model is not built. Run build method or load model before fitting')

        test_results = self.model.evaluate(x_test,
                                          y_test,
                                          batch_size=self.batch_size,
                                          verbose=verbose)
        self.val_history = test_results
        return test_results

    def get_layer_n_output(self, n, data):
        """Returns the output of layer n of the model for the given data"""

        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.layers[n].output)
        intermediate_output = intermediate_layer_model.predict(data)
        return intermediate_output

    def predict(self, data, verbose=0):
        """Model evaluation on a data set

            :returns : model predictions (numpy.nparray / shape: <n_samples, n_outputs>
        """

        if self.model is None:
            raise StandardError('Model is not built. Run build method or load model before fitting')
        return self.model.predict(data, 1, verbose)  # ,steps)

    def load(self, file_path):
        """Load model info and weights from a .h5 file"""
        self.model = load_model(file_path)

    def save(self, file_path):
        """Save current model state and weights on an .h5 file"""
        self.model.save(file_path)

    def purge(self):
        """Delete model state"""
        del self.model


class BaseNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass


    def plotTraining(self, ax=None,
                     train_scores='all',
                     val_scores='all',
                     savepath=None):
        if ax is None:
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(1,1)
            loss_ax = ax
            score_ax = plt.twinx(ax)
        history = self.history

        if val_scores is 'all':
            val_re = re.compile('val_')
            val_scores = map(lambda x: x.string,
                             filter(lambda x: x is not None,
                                    map(val_re.search, history.columns.values)))

        if train_scores is 'all':
            train_scores = history.columns.values[1:] # Remove 'epoch' column
            train_scores = train_scores[~np.isin(train_scores, val_scores)]

        x = history['epoch']
        linestyles = ['-', '--', '-.', ':']
        ls_train = cycle(linestyles)
        ls_val = cycle(linestyles)

        loss_finder = re.compile('loss')
        for train_score, val_score in zip(train_scores, val_scores):
            if loss_finder.search(train_score) is None:
                score_ax.plot(x, history[train_score], color="blue", linestyle=ls_train.next(), label=train_score)
            else:
                loss_ax.plot(x, history[train_score], color="blue", linestyle=ls_train.next(), label=train_score)

            if loss_finder.search(val_score) is None:
                score_ax.plot(x, history[val_score], color="red", linestyle=ls_val.next(), label=val_score)
            else:
                loss_ax.plot(x, history[val_score], color="red", linestyle=ls_val.next(), label=val_score)

        ax.legend()
        plt.show()

    def fit(self,
            X,
            y,
            n_inits = 1,
            validation_split=0.0,
            validation__data=None,
            shuffle=True,
            verbose=0,
            class_weight=True,
            sample_weight=None,
            steps_per_epoch=None,
            validation_steps=None):

        # scaler = StandardScaler()
        # self.scaler = scaler
        # scaler.fit(X=X, y=y)
        # X = scaler.transform(X)

        if class_weight:
            class_weights = self._getGradientWeights(y)
        else:
            class_weights = None

        if n_inits < 1:
            warnings.warn("Number of initializations must be at least one."
                          "Falling back to one")
            n_inits = 1
        # self.input_shape = X.shape[1:]


        # self.LofarObj.window_size = self.input_shape
        # self.LofarObj.fit(X,y)
        # X,y = self.LofarObj.transform(X)

        self.input_shape = X.shape[1:]
        print self.input_shape
        trnParams, callbacks, filepath = self._buildParameters()
        keras_callbacks = callbacks.toKerasFn()
        model = SequentialModelWrapper(trnParams,
                                       results_path=filepath)
        best_weigths_path = os.path.join(filepath, 'best_weights')

        if exists(best_weigths_path):
            print "Model trained, loading best weights"
            model.build_model()
        else:
            for init in range(n_inits):
                # for callback in keras_callbacks:
                #     if isinstance(callback, ModelCheckpoint):
                #         print callback.best

                model.build_model()
                print model.model.summary()

                # print model.model.optimizer.weights
                # before = model.model.get_weights()
                model.fit(x=X,
                          y=y,
                          batch_size=self.batch_size,
                          epochs=self.epochs,
                          verbose=verbose,
                          callbacks=keras_callbacks,
                          validation_split=validation_split,
                          validation_data=validation__data,
                          shuffle=shuffle,
                          class_weight=class_weights,
                          sample_weight=sample_weight,
                          initial_epoch=0,
                          steps_per_epoch=steps_per_epoch,
                          validation_steps=validation_steps)


                # print "After Training"
                # print model.model.get_weights()
                # print before == model.model.optimizer.weights
                # print model.model.optimizer.weights

        model.load_weights(best_weigths_path)
        self.history = pd.read_csv(os.path.join(filepath, 'history.csv'))
        self.model = model
        return self

    def predict(self, X):
        # X, y = self.LofarObj.transform(X)
        # Xt = self.scaler.transform(X)
        return self.model.predict(X)

    def score(self, X, y, sample_weight=None):
        if y.ndim > 1:
            y = y.argmax(axis=1)
        out = self.predict(X)
        # _, yt = self.LofarObj.transform(X)
        # yt = yt.argmax(axis=1)
        cat_out = out.argmax(axis=1)
        return spIndex(recall_score(y, cat_out),
                      n_classes=len(np.unique(y)))
        # return recall_score(y, cat_out)

    @staticmethod
    def _getGradientWeights(y_train, mode='standard'):
        if y_train.ndim > 1:
            y_train = y_train.argmax(axis=1)

        cls_indices, event_count = np.unique(np.array(y_train), return_counts=True)
        min_class = min(event_count)

        return {cls_index: float(min_class) / cls_count
                for cls_index, cls_count in zip(cls_indices, event_count)}

    @staticmethod
    def build_model_checkpoint(filepath,
                               monitor='val_loss',
                               verbose=0,
                               save_weights_only=False,
                               mode='auto',
                               period=1,
                               save_best=True):

        m_filepath = os.path.join(filepath, 'end_weights')
        b_filepath = os.path.join(filepath, 'best_weights')

        m_check =  {"type": "ModelCheckpoint",
                    "filepath": m_filepath,
                    "monitor": monitor,
                    "verbose": verbose,
                    "save_weights_only": save_weights_only,
                    "mode": mode,
                    "period": period}
        if save_best:
            best_check = {"type": "ModelCheckpoint",
                          "filepath": b_filepath,
                          "monitor": monitor,
                          "verbose" : verbose,
                          "save_weights_only": save_weights_only,
                          "mode" : mode,
                          "period" : period,
                          "save_best_only":True}
        else:
            best_check = None

        return m_check, best_check

    @staticmethod
    def build_conv_layer(filters,
                         kernel_size,
                         stride,
                         activation,
                         padding,
                         data_format,
                         dilation_rate,
                         kernel_initializer,
                         bias_initializer,
                         kernel_regularizer,
                         bias_regularizer,
                         activity_regularizer,
                         kernel_constraint,
                         bias_constraint,
                         input_shape = None):

        conv_dim = len(kernel_size)

        if len(stride) > conv_dim:
            raise NotImplementedError
        elif len(stride) < conv_dim and len(stride) != 1:
            raise NotImplementedError

        if not isinstance(filters, int):
            raise NotImplementedError
        if not isinstance(activation, str):
            raise NotImplementedError
        if not isinstance(padding, str):
            raise NotImplementedError
        if not isinstance(data_format, str):
            raise NotImplementedError

        if conv_dim == 1:
            type = "Conv1D"
        elif conv_dim == 2:
            type = "Conv2D"
        elif conv_dim == 3:
            type = "Conv3D"
        else:
            raise NotImplementedError
        # TODO handle other types of convolutions

        layer = {"type": type,
                 "filters": filters,
                 "kernel_size": kernel_size,
                 "strides": stride,
                 "padding": padding,
                 "data_format": data_format,
                 "dilation_rate": dilation_rate,
                 "activation": activation,
                 "kernel_initializer": kernel_initializer,
                 "bias_initializer": bias_initializer,
                 "kernel_regularizer": kernel_regularizer,
                 "bias_regularizer": bias_regularizer,
                 "activity_regularizer": activity_regularizer,
                 "kernel_constraint": kernel_constraint,
                 "bias_constraint": bias_constraint}

        return layer

    @staticmethod
    def build_pooling_layer(type,
                            pool_size,
                            stride,
                            padding,
                            data_format):

        pool_dim = len(pool_size)

        if len(stride) > pool_dim:
            raise NotImplementedError
        elif len(stride) < pool_dim and len(stride) != 1:
            raise NotImplementedError

        if not isinstance(padding, str):
            raise NotImplementedError

        # TODO handle other types of poolings
        if type not in ["MaxPooling", "AveragePooling"]:
            raise NotImplementedError
        if pool_dim == 1:
            type += "1D"
        elif pool_dim == 2:
            type += "2D"
        elif pool_dim == 3:
            type += "3D"

        layer = {"type": type,
                 "pool_size": pool_size,
                 "strides": stride,
                 "padding": padding,
                 "data_format": data_format}

        return layer

    @staticmethod
    def build_dense_layer(units,
                          activation,
                          kernel_initializer,
                          bias_initializer,
                          kernel_regularizer,
                          bias_regularizer,
                          activity_regularizer,
                          kernel_constraint,
                          bias_constraint,
                          input_shape=None):

        layer = {"type": "Dense",
                 "units": units,
                 "kernel_initializer": kernel_initializer,
                 "bias_initializer": bias_initializer,
                 "kernel_regularizer": kernel_regularizer,
                 "bias_regularizer": bias_regularizer,
                 "activity_regularizer": activity_regularizer,
                 "kernel_constraint": kernel_constraint,
                 "bias_constraint": bias_constraint}

        if input_shape is not None:
            layer["input_shape"] = input_shape

        if activation != "":
            layer["activation"] = activation
        return layer

    @staticmethod
    def build_optimizer(solver,
                        momentum=0.9,
                        nesterov=True,
                        decay=0.0,
                        learning_rate=0.001,
                        amsgrad=False,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-08):
        solver = solver.lower()

        optimizer = {}
        if solver not in ['sgd', 'adam']:
            raise NotImplementedError

        if solver == 'sgd':
            optimizer = {"type": "SGD",
                         "momentum": momentum,
                         "decay": decay,
                         "nesterov": nesterov}

        elif solver == 'adam':
            optimizer = {"type": "Adam",
                         "lr": learning_rate,
                         "beta_1": beta_1,
                         "beta_2": beta_2,
                         "epsilon": epsilon,
                         "decay": decay,
                         "amsgrad": amsgrad}

        return optimizer


    @staticmethod
    def build_early_stopping(monitor,
                             min_delta,
                             patience,
                             verbose,
                             mode,
                             baseline,
                             restore_best_weights):
        return  {"type": "EarlyStopping",
                 "monitor":monitor,
                 "min_delta":min_delta,
                 "patience":patience,
                 "verbose":verbose,
                 "mode":mode,
                 "baseline":baseline,
                 "restore_best_weights":restore_best_weights}


class ConvNetClassifier(BaseNNClassifier):
    def __init__(self,
                 LofarObj = None,
                 input_shape=(None,),
                 n_filters=(6,),
                 conv_filter_sizes=((4, 4),),
                 conv_strides=((1, 1),),
                 pool_filter_sizes=((2, 2),),
                 pool_strides=((1, 1),),
                 conv_activations=("relu",),
                 conv_padding=('valid',),
                 conv_dropout=None,
                 pool_padding=('valid',),
                 pool_types=('MaxPooling',),
                 conv_dilation_rate=(1,),
                 pool_dilation_rate=(1,),
                 data_format = 'channels_last',
                 dense_layer_sizes=(10,),
                 dense_activations=("relu",),
                 dense_dropout=None,
                 solver="adam",
                 batch_size=32,
                 epochs=200,
                 loss="categorical_crossentropy",
                 metrics=["acc"],
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 momentum=0.9,
                 nesterov=True,
                 decay=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-08,
                 learning_rate=0.001,
                 amsgrad=False,
                 early_stopping=False,
                 es_kwargs=None,
                 model_checkpoint=True,
                 save_best=True,
                 mc_kwargs=None,
                 log_history=True,
                 cachedir='./'):

        self.cachedir = cachedir
        self.input_shape = input_shape
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        self.LofarObj = LofarObj
        for arg, val in values.items():
            setattr(self, arg, val)

    def _buildParameters(self):
        if self.es_kwargs is None:
            es_kwargs = {"monitor": 'val_loss',
                         "min_delta": 0,
                         "patience": 10,
                         "verbose": 0,
                         "mode": 'auto',
                         "baseline": None,
                         "restore_best_weights": False}
        else:
            tmp_kwargs = {"monitor": 'val_loss',
                          "min_delta": 0,
                          "patience": 10,
                          "verbose": 0,
                          "mode": 'auto',
                          "baseline": None,
                          "restore_best_weights": False}

            for key in self.es_kwargs.keys():
                tmp_kwargs[key] = self.es_kwargs[key]
            es_kwargs = tmp_kwargs

        if self.mc_kwargs is None:
            mc_kwargs = {"monitor": 'val_loss',
                         "verbose": 0,
                         "save_weights_only": False,
                         "mode": 'auto',
                         "period": 1,
                         "save_best":self.save_best}
        else:
            tmp_kwargs = {"monitor": 'val_loss',
                         "verbose": 0,
                         "save_weights_only": False,
                         "mode": 'auto',
                         "period": 1,
                         "save_best": self.save_best}
            for key in self.mc_kwargs.keys():
                tmp_kwargs[key] = self.mc_kwargs[key]
            mc_kwargs = tmp_kwargs


        conv_layers = [self.build_conv_layer(filters,
                                             kernel_size,
                                             stride,
                                             activation,
                                             padding,
                                             self.data_format,
                                             dilation_rate,
                                             self.kernel_initializer,
                                             self.bias_initializer,
                                             self.kernel_regularizer,
                                             self.bias_regularizer,
                                             self.activity_regularizer,
                                             self.kernel_constraint,
                                             self.bias_constraint)
                       for filters, kernel_size, stride, activation, padding, dilation_rate
                       in zip(self.n_filters,
                              self.conv_filter_sizes,
                              self.conv_strides,
                              self.conv_activations,
                              self.conv_padding,
                              self.conv_dilation_rate)
                       ]

        pool_layers = [self.build_pooling_layer(pool_type,
                                                pool_size,
                                                stride,
                                                padding,
                                                self.data_format)
                       for pool_type, pool_size, stride, padding
                       in zip(self.pool_types,
                              self.pool_filter_sizes,
                              self.pool_strides,
                              self.pool_padding)
                       ]

        def intercalate(iter1, iter2):
            for el1, el2 in zip(iter1, iter2):
                yield el1
                yield el2

        conv_pool_layers = list(intercalate(conv_layers, pool_layers))

        conv_pool_layers.append({"type":"Flatten"})

        dense_layers = [self.build_dense_layer(units,
                                               activation,
                                               self.kernel_initializer,
                                               self.bias_initializer,
                                               self.kernel_regularizer,
                                               self.bias_regularizer,
                                               self.activity_regularizer,
                                               self.kernel_constraint,
                                               self.bias_constraint)
                        for units, activation in zip(self.dense_layer_sizes, self.dense_activations)]

        if self.dense_dropout is not None:
            dropout_pos = {pos+1: keras.layers.Dropout(rate)
                           for pos, rate in enumerate(self.dense_dropout) if rate is not None}
            for pos, drop_layer in dropout_pos.items():
                dense_layers.insert(pos, drop_layer)

        layers = np.concatenate([conv_pool_layers, dense_layers])

        layers[0]["input_shape"] = self.input_shape


        optimizer = self.build_optimizer(self.solver,
                                         self.momentum,
                                         self.nesterov,
                                         self.decay,
                                         self.learning_rate,
                                         self.amsgrad,
                                         self.beta_1,
                                         self.beta_2,
                                         self.epsilon)

        trnParams = CNNParams(prefix='convnet',
                              optimizer=optimizer,
                              layers=layers,
                              loss=self.loss,
                              metrics=self.metrics,
                              epochs=self.epochs,
                              batch_size=self.batch_size,
                              callbacks=None,
                              input_shape=self.input_shape)

        filepath = os.path.join(self.cachedir, trnParams.getParamPath())

        mc_kwargs["filepath"] = filepath
        callbacks = []
        if self.early_stopping:
            callbacks.append(self.build_early_stopping(**es_kwargs))
        if self.model_checkpoint:
            m_check, best_model = self.build_model_checkpoint(**mc_kwargs)
            callbacks.append(m_check)
            if best_model is not None:
                callbacks.append(best_model)
        if self.log_history:
            csvlog = {"type": "CSVLogger", "filename": os.path.join(filepath, 'history.csv')}
            callbacks.append(csvlog)

        callbacks_list = TrainParameters.Callbacks()
        callbacks_list.add(callbacks)
        return trnParams, callbacks_list, filepath



class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 hidden_layer_sizes=(10,),
                 activations=("relu",),
                 solver="adam",
                 batch_size=32,
                 epochs=200,
                 loss="categorical_crossentropy",
                 metrics=["acc"],
                 input_shape=(None,),
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 momentum=0.9,
                 nesterov=True,
                 decay=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-08,
                 learning_rate=0.001,
                 amsgrad=False,
                 early_stopping=False,
                 es_kwargs=None,
                 model_checkpoint=True,
                 save_best=True,
                 mc_kwargs=None,
                 log_history=True,
                 cachedir='./'):

        self.cachedir = cachedir
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)


    def _buildParameters(self):

        if self.es_kwargs is None:
            es_kwargs = {"monitor": 'val_loss',
                         "min_delta": 0,
                         "patience": 10,
                         "verbose": 0,
                         "mode": 'auto',
                         "baseline": None,
                         "restore_best_weights": False}
        else:
            tmp_kwargs = {"monitor": 'val_loss',
                          "min_delta": 0,
                          "patience": 10,
                          "verbose": 0,
                          "mode": 'auto',
                          "baseline": None,
                          "restore_best_weights": False}

            for key in self.es_kwargs.keys():
                tmp_kwargs[key] = self.es_kwargs[key]
            es_kwargs = tmp_kwargs

        if self.mc_kwargs is None:
            mc_kwargs = {"monitor": 'val_loss',
                         "verbose": 0,
                         "save_weights_only": False,
                         "mode": 'auto',
                         "period": 1,
                         "save_best": self.save_best}
        else:
            tmp_kwargs = {"monitor": 'val_loss',
                         "verbose": 0,
                         "save_weights_only": False,
                         "mode": 'auto',
                         "period": 1,
                         "save_best": self.save_best}
            for key in self.mc_kwargs.keys():
                tmp_kwargs[key] = self.mc_kwargs[key]
            mc_kwargs = tmp_kwargs

        layers = [self.build_layer(units,
                                   activation,
                                   self.kernel_initializer,
                                   self.bias_initializer,
                                   self.kernel_regularizer,
                                   self.bias_regularizer,
                                   self.activity_regularizer,
                                   self.kernel_constraint,
                                   self.bias_constraint)
                  for units, activation in zip(self.hidden_layer_sizes, self.activations)]
        layers[0] = self.build_layer(self.hidden_layer_sizes[0],
                                     self.activations[0],
                                     self.kernel_initializer,
                                     self.bias_initializer,
                                     self.kernel_regularizer,
                                     self.bias_regularizer,
                                     self.activity_regularizer,
                                     self.kernel_constraint,
                                     self.bias_constraint,
                                     input_shape=self.input_shape)


        optimizer = self.build_optimizer(self.solver,
                                         self.momentum,
                                         self.nesterov,
                                         self.decay,
                                         self.learning_rate,
                                         self.amsgrad,
                                         self.beta_1,
                                         self.beta_2,
                                         self.epsilon)

        trnParams = CNNParams(prefix='mlp',
                              optimizer=optimizer,
                              layers=layers,
                              loss=self.loss,
                              metrics=self.metrics,
                              epochs=self.epochs,
                              batch_size=self.batch_size,
                              callbacks=None,
                              input_shape=self.input_shape)

        filepath = os.path.join(self.cachedir, trnParams.getParamPath())

        mc_kwargs["filepath"] = filepath
        callbacks = []
        if self.early_stopping:
            callbacks.append(self.build_early_stopping(**es_kwargs))
        if self.model_checkpoint:
            m_check, best_model = self.build_model_checkpoint(**mc_kwargs)
            callbacks.append(m_check)
            if best_model is not None:
                callbacks.append(best_model)
        if self.log_history:
            csvlog = {"type": "CSVLogger", "filename": os.path.join(filepath, 'history.csv')}
            callbacks.append(csvlog)

        callbacks_list = TrainParameters.Callbacks()
        callbacks_list.add(callbacks)
        return trnParams, callbacks_list, filepath

    def plotTraining(self, ax=None,
                     train_scores='all',
                     val_scores='all',
                     savepath=None):
        if ax is None:
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(1,1)
            loss_ax = ax
            score_ax = plt.twinx(ax)
        history = self.history

        if val_scores is 'all':
            val_re = re.compile('val_')
            val_scores = map(lambda x: x.string,
                             filter(lambda x: x is not None,
                                    map(val_re.search, history.columns.values)))

        if train_scores is 'all':
            train_scores = history.columns.values[1:] # Remove 'epoch' column
            train_scores = train_scores[~np.isin(train_scores, val_scores)]

        x = history['epoch']
        linestyles = ['-', '--', '-.', ':']
        ls_train = cycle(linestyles)
        ls_val = cycle(linestyles)

        loss_finder = re.compile('loss')
        for train_score, val_score in zip(train_scores, val_scores):
            if loss_finder.search(train_score) is None:
                score_ax.plot(x, history[train_score], color="blue", linestyle=ls_train.next(), label=train_score)
            else:
                loss_ax.plot(x, history[train_score], color="blue", linestyle=ls_train.next(), label=train_score)

            if loss_finder.search(val_score) is None:
                score_ax.plot(x, history[val_score], color="red", linestyle=ls_val.next(), label=val_score)
            else:
                loss_ax.plot(x, history[val_score], color="red", linestyle=ls_val.next(), label=val_score)

        ax.legend()
        plt.show()

    def fit(self,
            X,
            y,
            n_inits = 1,
            validation_split=0.0,
            validation__data=None,
            shuffle=True,
            verbose=0,
            class_weight=True,
            sample_weight=None,
            steps_per_epoch=None,
            validation_steps=None):

        if class_weight:
            class_weights = self._getGradientWeights(y)
        else:
            class_weights = None

        if n_inits < 1:
            warnings.warn("Number of initializations must be at least one."
                          "Falling back to one")
            n_inits = 1

        print self.input_shape

        trnParams, callbacks, filepath = self._buildParameters()
        keras_callbacks = callbacks.toKerasFn()
        model = SequentialModelWrapper(trnParams,
                                       results_path=filepath)
        best_weigths_path = os.path.join(filepath, 'best_weights')

        if exists(best_weigths_path):
            print "Model trained, loading best weights"
            model.build_model()
        else:
            for init in range(n_inits):
                # for callback in keras_callbacks:
                #     if isinstance(callback, ModelCheckpoint):
                #         print callback.best

                model.build_model()

                # print model.model.optimizer.weights
                before = model.model.get_weights()
                model.fit(x=X,
                          y=y,
                          batch_size=self.batch_size,
                          epochs=self.epochs,
                          verbose=verbose,
                          callbacks=keras_callbacks,
                          validation_split=validation_split,
                          validation_data=validation__data,
                          shuffle=shuffle,
                          class_weight=class_weights,
                          sample_weight=sample_weight,
                          initial_epoch=0,
                          steps_per_epoch=steps_per_epoch,
                          validation_steps=validation_steps)

            import keras.backend as K
            K.clear_session()
                # print "After Training"
                # print model.model.get_weights()
                # print before == model.model.optimizer.weights
                # print model.model.optimizer.weights

        model.load_weights(best_weigths_path)
        self.history = pd.read_csv(os.path.join(filepath, 'history.csv'))
        self.model = model
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y, sample_weight=None):
        if y.ndim > 1:
            y = y.argmax(axis=1)

        out = self.predict(X)

        cat_out = out.argmax(axis=1)
        return spIndex(recall_score(y, cat_out),
                       n_classes=len(np.unique(y)))

    @staticmethod
    def _getGradientWeights(y_train, mode='standard'):
        if y_train.ndim > 1:
            y_train = y_train.argmax(axis=1)

        cls_indices, event_count = np.unique(np.array(y_train), return_counts=True)
        min_class = min(event_count)

        return {cls_index: float(min_class) / cls_count
                for cls_index, cls_count in zip(cls_indices, event_count)}

    @staticmethod
    def build_model_checkpoint(filepath,
                               monitor='val_loss',
                               verbose=0,
                               save_weights_only=False,
                               mode='auto',
                               period=1,
                               save_best=True):

        m_filepath = os.path.join(filepath, 'end_weights')
        b_filepath = os.path.join(filepath, 'best_weights')

        m_check =  {"type": "ModelCheckpoint",
                    "filepath": m_filepath,
                    "monitor": monitor,
                    "verbose": verbose,
                    "save_weights_only": save_weights_only,
                    "mode": mode,
                    "period": period}
        if save_best:
            best_check = {"type": "ModelCheckpoint",
                          "filepath": b_filepath,
                          "monitor": monitor,
                          "verbose" : verbose,
                          "save_weights_only": save_weights_only,
                          "mode" : mode,
                          "period" : period,
                          "save_best_only":True}
        else:
            best_check = None

        return m_check, best_check

    @staticmethod
    def build_early_stopping(monitor,
                             min_delta,
                             patience,
                             verbose,
                             mode,
                             baseline,
                             restore_best_weights):
        return  {"type": "EarlyStopping",
                 "monitor":monitor,
                 "min_delta":min_delta,
                 "patience":patience,
                 "verbose":verbose,
                 "mode":mode,
                 "baseline":baseline,
                 "restore_best_weights":restore_best_weights}

    @staticmethod
    def build_layer(units,
                    activation,
                    kernel_initializer,
                    bias_initializer,
                    kernel_regularizer,
                    bias_regularizer,
                    activity_regularizer,
                    kernel_constraint,
                    bias_constraint,
                    input_shape=None):

        layer = {"type": "Dense",
                 "units": units,
                 "kernel_initializer": kernel_initializer,
                 "bias_initializer": bias_initializer,
                 "kernel_regularizer": kernel_regularizer,
                 "bias_regularizer": bias_regularizer,
                 "activity_regularizer": activity_regularizer,
                 "kernel_constraint": kernel_constraint,
                 "bias_constraint": bias_constraint}

        if input_shape is not None:
            layer["input_shape"] = input_shape

        if activation != "":
            layer["activation"] = activation
        return layer

    @staticmethod
    def build_optimizer(solver,
                        momentum=0.9,
                        nesterov=True,
                        decay=0.0,
                        learning_rate=0.001,
                        amsgrad=False,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-08):
        solver = solver.lower()

        optimizer = {}
        if solver not in ['sgd', 'adam']:
            raise NotImplementedError

        if solver == 'sgd':
            optimizer = {"type": "SGD",
                         "momentum": momentum,
                         "decay": decay,
                         "nesterov": nesterov}

        elif solver == 'adam':
            optimizer = {"type": "Adam",
                         "lr": learning_rate,
                         "beta_1": beta_1,
                         "beta_2": beta_2,
                         "epsilon": epsilon,
                         "decay": decay,
                         "amsgrad": amsgrad}

        return optimizer


class SequentialModelWrapper():
    """Keras Sequential Model wrapper class

       Inherits from _CNNModel and adds details
       for interfacing with the Keras API
    """

    def __init__(self, trnParams, results_path):
        #super(SequentialModelWrapper, self).__init__()
        self.path = ModelPaths(trnParams, results_path)
        self._mountParams(trnParams)
        if not SystemIO.exists(results_path + '/' + trnParams.getParamPath()):
            self.createFolders(results_path + '/' + trnParams.getParamPath())
            self.saveParams(trnParams)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def _mountParams(self, trnParams):
        """Parses the parameters to attributes of the instance

           trnParams (TrnParamsConvolutional): parameter object
        """
        for param, value in trnParams:
            if param is None:
                raise ValueError('The parameters configuration received by _CNNModel must be complete'
                                 '%s was passed as NoneType.'
                                 % (param))
            setattr(self, param, value)

    def createFolders(self, *args):
        """Creates model folder structure from hyperparameters information"""
        for folder in args:
            mkdir(folder)

    def saveParams(self, trnParams):
        """Save parameters into a pickle file"""

        SystemIO.save(trnParams.toNpArray(), self.path.model_info_file)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __call__(self, *args, **kwargs):
        self.__init__(*args)

    def build_model(self):
        """Compile Keras model using the parameters loaded into the instance"""
        self.model = Sequential()
        print self.layers[0].identifier
        print self.layers[0].parameters
        for layer in self.layers:
            # print layer.identifier
            # print layer.parameters
            self.model.add(layer.toKerasFn())


        # super(SequentialModelWrapper, self).compile(optimizer=self.optimizer.toKerasFn(),
        #                                             loss=self.loss,
        #                                             metrics=self.metrics)
        self.model.compile(optimizer=self.optimizer.toKerasFn(),
                           loss=self.loss,
                           metrics=self.metrics)

    def fit(self, *args, **kwargs):
        # for arg in args:
        #     print arg.shape
        return self.model.fit(*args, **kwargs)

        # def fit(x=None,
    #         y=None,
    #         batch_size=None,
    #         epochs=1,
    #         verbose=1,
    #         callbacks=None,
    #         validation_split=0.0,
    #         validation_data=None,
    #         shuffle=True,
    #         class_weight=None,
    #         sample_weight=None,
    #         initial_epoch=0,
    #         steps_per_epoch=None,
    #         validation_steps=None):
    #     """Model training routine
    #
    #        x_train (numpy.nparray): model input data
    #        y_train (numpy.nparray): input data labels
    #        callbacks: manual callbacks insertion. It will be removed on future commits
    #        validation_data (<numpy.nparray, numpy.nparray>): tuple containing input and trgt
    #                                                          data for model validation. <input, trgt>
    #        class_weight (dict): dictionary mapping class indices to a floating point value,
    #                             used for weighting the loss function during training
    #                             (see ConvolutionTrainFunction.getClassWeights)
    #
    #        max_restarts (int): max number of restarts if the desired conditions are not reached
    #        restart_tol: tolerance value for the restart condition
    #        restart_monitor: condition for restarting training
    #
    #         :returns : dictionary containing loss and metrics for each epoch
    #     """
    #
    #     if self.model is None:
    #         raise StandardError('Model is not built. Run build method or load model before fitting')
    #
    #     #restarts_values = np.zeros(n_restarts)
    #
    #     # for i_restart in range(n_restarts):
    #     history = self.model.fit(x=None,
    #                              y=None,
    #                              batch_size=None,
    #                              epochs=1,
    #                              verbose=1,
    #                              callbacks=None,
    #                              validation_split=0.0,
    #                              validation_data=None,
    #                              shuffle=True,
    #                              class_weight=None,
    #                              sample_weight=None,
    #                              initial_epoch=0,
    #                              steps_per_epoch=None,
    #                              validation_steps=None)
    #         # restarts_values[i_restart] = min(history.history['val_loss'])
    #     return history


    def evaluate(self, x_test, y_test, verbose=0):
        """Model evaluation on a test set

            x_test: input data
            y_test: data labels

            :returns : evaluation results
        """

        if self.model is None:
            raise StandardError('Model is not built. Run build method or load model before fitting')

        test_results = self.model.evaluate(x_test,
                                          y_test,
                                          batch_size=self.batch_size,
                                          verbose=verbose)
        self.val_history = test_results
        return test_results

    def get_layer_n_output(self, n, data):
        """Returns the output of layer n of the model for the given data"""

        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.layers[n].output)
        intermediate_output = intermediate_layer_model.predict(data)
        return intermediate_output

    def predict(self, data, verbose=0):
        """Model evaluation on a data set

            :returns : model predictions (numpy.nparray / shape: <n_samples, n_outputs>
        """

        if self.model is None:
            raise StandardError('Model is not built. Run build method or load model before fitting')
        # print len(data[0])
        # print type(data[0])
        # print data.shape
        return self.model.predict(data, 1, verbose)  # ,steps)

    def load(self, file_path):
        """Load model info and weights from a .h5 file"""
        self.model = load_model(file_path)

    def save(self, file_path):
        """Save current model state and weights on an .h5 file"""
        self.model.save(file_path)

    def purge(self):
        """Delete model state"""
        del self.model