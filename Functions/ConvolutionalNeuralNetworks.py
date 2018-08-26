'''
   Author: Pedro Henrique Braga Lisboa pedrolisboa at poli.ufrj.br
   This module gather utilities to implement a Convolutional Neural Network
   for Classification and Novelty Detection.
'''

import os
import warnings
from abc import abstractmethod
from warnings import warn

import numpy as np
from keras import Sequential, Model
from keras.models import load_model

from Functions import SystemIO
from Functions.SystemIO import load, exists, mkdir
from Functions.TrainParameters import TrnParamsConvolutional


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
    #results_path = './Analysis'
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
    def __init__(self, trnParams):
        super(ModelPaths, self).__init__()
        if not isinstance(trnParams, TrnParamsConvolutional):
            raise ValueError('The parameters inserted must be an instance of TrnPararamsConvolutional'
                             '%s of type %s was passed.'
                             % (trnParams, type(trnParams)))

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
        self.model_history = self.fold_path + '/' + 'history'
        self.model_recovery_history = self.fold_path + '/' + 'history/~history.npy'

        self.model_predictions = self.fold_path + '/' + 'predictions'
        self.model_recovery_predictions = self.fold_path + '/' + 'predictions/~predictions.npy'

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
            self.createFolders(self.model_files, self.model_best, self.model_history, self.model_predictions)
            self.status = 'Untrained'

        self.trained_folds_files = os.listdir(self.model_files)

    def createFolders(self, *args):
        """Creates model folder structure from hyperparameters information"""
        for folder in args:
            mkdir(folder)

    def purgeFolder(self, subfolder=''):
        raise NotImplementedError


class _CNNModel(ModelPaths):
    def __init__(self, trnParams):
        super(_CNNModel, self).__init__(trnParams)
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

    def fit(self, x_train, y_train, callbacks, validation_data=None, class_weight=None, verbose=0):
        # TODO fix method signature removing manual callback selection
        if self.model is None:
            raise StandardError('Model is not built. Run build method or load model before fitting')

        history = self.model.fit(x_train,
                                 y_train,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 callbacks=callbacks,
                                 validation_data=validation_data,
                                 class_weight=class_weight,
                                 verbose=verbose
                                 )
        self.history = history.history
        return history.history

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