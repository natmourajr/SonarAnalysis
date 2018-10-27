"""
    This file contents some classification analysis functions
"""

import os
import keras
import numpy as np

from collections import OrderedDict
from warnings import warn

from keras.utils import get_custom_objects
from sklearn.externals import joblib
from sklearn import model_selection

from Functions.TfUtils.Scores import spIndex
from Functions.SystemIO import load


class TrnParams(object):
    """
        Basic class
    """

    def __init__(self, analysis="None"):
        self.analysis = analysis
        self.params = None

    def save(self, name="None"):
        joblib.dump([self.params], name, compress=9)

    def load(self, name="None"):
        [self.params] = joblib.load(name)

    def printParams(self):
        for iparameter in self.params:
            print iparameter + ': ' + str(self.params[iparameter])


# classification

def ClassificationFolds(folder, n_folds=2, trgt=None, dev=False, verbose=False):
    if n_folds < 2:
        print 'Invalid number of folds'
        return -1

    if not dev:
        file_name = '%s/%i_folds_cross_validation.jbl' % (folder, n_folds)
    else:
        file_name = '%s/%i_folds_cross_validation_dev.jbl' % (folder, n_folds)

    if not os.path.exists(file_name):
        if verbose:
            print "Creating %s" % (file_name)

        if trgt is None:
            print 'Invalid trgt'
            return -1

        CVO = model_selection.StratifiedKFold(trgt, n_folds)
        CVO = list(CVO)
        joblib.dump([CVO], file_name, compress=9)
    else:
        if verbose:
            print "File %s exists" % (file_name)
        [CVO] = joblib.load(file_name)

    return CVO


class NeuralClassificationTrnParams(TrnParams):
    """
        Neural Classification TrnParams
    """

    def __init__(self,
                 n_inits=2,
                 norm='mapstd',
                 verbose=False,
                 train_verbose=False,
                 n_epochs=10,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-08,
                 learning_decay=1e-6,
                 momentum=0.3,
                 nesterov=True,
                 patience=5,
                 batch_size=4,
                 hidden_activation='tanh',
                 output_activation='tanh',
                 metrics=['accuracy'],
                 loss='mean_squared_error',
                 optmizerAlgorithm='SGD'
                 ):
        self.params = {}

        self.params['n_inits'] = n_inits
        self.params['norm'] = norm
        self.params['verbose'] = verbose
        self.params['train_verbose'] = train_verbose

        # train params
        self.params['n_epochs'] = n_epochs
        self.params['learning_rate'] = learning_rate
        self.params['beta_1'] = beta_1
        self.params['beta_2'] = beta_2
        self.params['epsilon'] = epsilon
        self.params['learning_decay'] = learning_decay
        self.params['momentum'] = momentum
        self.params['nesterov'] = nesterov
        self.params['patience'] = patience
        self.params['batch_size'] = batch_size
        self.params['hidden_activation'] = hidden_activation
        self.params['output_activation'] = output_activation
        self.params['metrics'] = metrics
        self.params['loss'] = loss
        self.params['optmizerAlgorithm'] = optmizerAlgorithm

    def get_params_str(self):
        param_str = ('%i_inits_%s_norm_%i_epochs_%i_batch_size_%s_hidden_activation_%s_output_activation' %
                     (self.params['n_inits'], self.params['norm'], self.params['n_epochs'], self.params['batch_size'],
                      self.params['hidden_activation'], self.params['output_activation']))
        for imetric in self.params['metrics']:
            param_str = param_str + '_' + imetric

        param_str = param_str + '_metric_' + self.params['loss'] + '_loss'
        return param_str


# novelty detection

def NoveltyDetectionFolds(folder, n_folds=2, trgt=None, dev=False, verbose=False):
    if n_folds < 2:
        print 'Invalid number of folds'
        return -1

    if not dev:
        file_name = '%s/%i_folds_cross_validation.jbl' % (folder, n_folds)
    else:
        file_name = '%s/%i_folds_cross_validation_dev.jbl' % (folder, n_folds)

    if not os.path.exists(file_name):
        if verbose:
            print "Creating %s" % (file_name)

        if trgt is None:
            print 'Invalid trgt'
            return -1

        CVO = {}
        for inovelty, novelty_class in enumerate(np.unique(trgt)):
            skf = model_selection.StratifiedKFold(n_splits=n_folds)
            process_trgt = trgt[trgt != novelty_class]
            CVO[inovelty] = skf.split(X=np.zeros(process_trgt.shape), y=process_trgt)
            CVO[inovelty] = list(CVO[inovelty])
        if verbose:
            print 'Saving in %s' % (file_name)

        joblib.dump([CVO], file_name, compress=9)

    else:
        if verbose:
            print "Reading from %s" % (file_name)

        [CVO] = joblib.load(file_name)

    return CVO


class SVMNoveltyDetectionTrnParams(TrnParams):
    """
        SVM Novelty Detection TrnParams
    """

    def __init__(self,
                 norm='mapstd',
                 gamma=0.01,
                 kernel='rbf',
                 verbose=False):
        self.params = {}

        self.params['norm'] = norm
        self.params['verbose'] = verbose
        self.params['gamma'] = gamma
        self.params['kernel'] = kernel

    def get_params_str(self):
        gamma_str = ('%1.5f' % (self.params['gamma'])).replace('.', '_')
        param_str = ('%s_norm_%s_gamma_%s_kernel' %
                     (self.params['norm'], gamma_str, self.params['kernel']))
        return param_str


class NNNoveltyDetectionTrnParams(NeuralClassificationTrnParams):
    """
        NN Novelty Detection TrnParams
    """


class SAENoveltyDetectionTrnParams(NeuralClassificationTrnParams):
    """
        SAE Novelty Detection TrnParams
    """

class TrnParamsConvolutional(object):
    def __init__(self,
                 prefix="convnet",
                 optimizer= None,
                 layers= None,
                 loss=None,
                 metrics=None,
                 epochs=10,
                 batch_size=32,
                 callbacks=[],
                 scale=False,
                 input_shape=(28, 28, 1)
                 ):

        # Default parameter settings -------------------------------
        if optimizer is None:
            optimizer = ["SGD", {'lr': 0.01,
                                    'decay': 1e-6,
                                    'momentum': 0.9,
                                    'nesterov': True}]
        if layers is None:
            layers = [["Conv2D", {"filters": 6,
                                  "kernel_size": (4, 4),
                                  "strides": 1,
                                  "data_format": "channels_last",
                                  "padding": "same"
                                 }
                      ],
                      ["Activation", {"activation": "relu"}],
                      ["MaxPooling2D", {"pool_size": (2, 2),
                                       "padding": "valid",
                                       "strides": None}
                      ],
                      ["Flatten", {}],
                      ["Dense", {"units": 50}],
                      ["Activation", {"activation": "tanh"}],
                      ["Dense", {"units": 4}],
                      ["Activation", {"activation": "softmax"}]]
        if loss is None:
            loss = "mean_squared_error"

        if metrics is None:
            metrics = ['acc', spIndex]

        get_custom_objects().update({"spIndex": spIndex})

        # Place input shape on first layer parameter list
        layers[0][1]['input_shape'] = input_shape

        # Setting parameters --------------------------------------
        self.__dict__ = OrderedDict()
        self.__dict__['input_shape'] = input_shape
        self.__dict__['prefix'] = prefix
        self.list2Optimizer(optimizer)
        self.list2layers(layers)
        self.__dict__['loss'] = loss[0] if isinstance(loss, list) else loss
        self.__dict__['metrics'] = metrics
        self.__dict__['epochs'] = epochs
        self.__dict__['batch_size'] = batch_size
        self.__dict__['scale'] = scale

        if isinstance(callbacks, Callbacks) or callbacks is None:
            self.__dict__['callbacks'] = Callbacks
        elif isinstance(layers, list):
            self.__dict__['callbacks'] = Callbacks(callbacks)
        else:
            raise ValueError('layers must be an instance of Layers or list'
                             '%s of type %s was passed' % (callbacks, type(callbacks)))

    @property
    def inputShape(self):
        return self.layers[0].parameters['input_shape']

    def toJson(self):
        return {key: value if not (isinstance(value, Parameter) or isinstance(value, _ParameterSet))
        else value.toJson() for key, value in self}


    def changeLayer(self, layer_name, new_layer):
        #TODO implement error checking

        for i,layer in enumerate(self.layers):
            if layer.identifier == layer_name:
                self.layers[i] = new_layer

    def changeLayerParams(self, layer_name, param_name, new_value):
        # TODO implement error checking

        for i,layer in enumerate(self.layers):
            if layer.identifier == layer_name:
                for parameter_key in layer.parameters:
                    if parameter_key == param_name:

                        self.layers[i].parameters[param_name] = new_value

    def list2Optimizer(self, optimizer):
        if isinstance(optimizer, Optimizer) or optimizer is None:
            self.__dict__['optimizer'] = optimizer
        elif isinstance(optimizer, list):
            new_optimizer = optimizer[1]
            new_optimizer["type"] = optimizer[0]
            self.__dict__['optimizer'] = Optimizer(new_optimizer)
        else:
            raise ValueError('optimizer must be an instance of Optimizer or list'
                             '%s of type %s was passed' % (optimizer, type(optimizer)))

    def list2layers(self, layers):
        if isinstance(layers, Layers) or layers is None:
            self.__dict__['layers'] = layers
        elif isinstance(layers, list):
            self.__dict__['layers'] = Layers(layers)
        else:
            raise ValueError('layers must be an instance of Layers or list'
                             '%s of type %s was passed' % (layers, type(layers)))

    @classmethod
    def fromfile(cls, filepath):
        params = load(filepath + '/model_info.jbl')
        return cls(*params)

    def __getitem__(self, param):
        return self.__dict__[param]

    def __getattr__(self, param):
        return self.__dict__[param]

    def __iter__(self):
        return self.next()

    def _storeParamObj(self, param, param_type, param_key, error_str):
        raise NotImplementedError

    def next(self):
        for param in self.__dict__:
            yield (param, self.__dict__[param])
        raise StopIteration

    def getParamPath(self):
        path = self.prefix + '/' + self.optimizer.getPathStr() + '/'
        for layer in self.layers:
            if layer.identifier == 'Activation':
                path = path + layer.parameters['activation'] + '_'
            else:
                path = path + layer.identifier + '_'
        path = path[:-1] + '/'  # remove last '_'
        for layer in self.layers:
            path = path + layer.getPathStr()
        path = path[:-1] + '/' + self.loss
        for metric in self.metrics:
            str_metric = metric if isinstance(metric, str) else metric.__name__
            path = path + '_' + str_metric
        path = path + '_' + str(self.epochs)
        path = path + '_' + str(self.batch_size)
        path = path + '_' + str(self.scale)

        return path

    def fitMask(self, paramMask):
        for (mask_param, mask_value) in paramMask:
            if isinstance(mask_value, _ParameterSet) or isinstance(mask_value, Parameter):
                if not self[mask_param].fitMask(mask_value):
                    return False
            else:
                if not mask_value == self[mask_param]:
                    return False
        return True

    def pprint(self):
        raise NotImplementedError

    def toNpArray(self):
        if self.callbacks is not None:
            cbks = self.callbacks.toNpArray() # Compatibility purposes
        else:
            cbks = None

        return [self.prefix,
                self.optimizer.toNpArray(),
                self.layers.toNpArray(),
                self.loss,
                self.metrics,
                self.epochs,
                self.batch_size,
                cbks,
                self.scale,
                self.layers.toNpArray()[0][1]['input_shape']
                ]


class CNNParams(object):
    def __init__(self,
                 prefix=None,
                 optimizer= None,
                 layers= None,
                 loss=None,
                 metrics=None,
                 epochs=10,
                 batch_size=32,
                 callbacks=None,
                 input_shape=(40, 400, 1)
                 ):

        # Default parameter settings -------------------------------
        if optimizer is None:
            optimizer = {"SGD": {'lr': 0.01,
                                 'decay': 1e-6,
                                 'momentum': 0.9,
                                 'nesterov': True}}

        if loss is None:
            loss = "mean_squared_error"

        if metrics is None:
            metrics = ['acc', spIndex]

        get_custom_objects().update({"spIndex": spIndex})

        # Place input shape on first layer parameter list
        #layers[0][1]['input_shape'] = input_shape

        # Setting parameters --------------------------------------
        self.__dict__ = OrderedDict()
        self.__dict__['input_shape'] = input_shape
        self.__dict__['prefix'] = prefix

        self.__dict__['optimizer'] = Optimizer(optimizer)

        self.__dict__['layers'] = Layers()

        if layers is not None:
            for layer in layers:
                # name = layer['type']
                # args = layer
                # del args['type']
                self.layers.add(layer)

        self.__dict__['callbacks'] = callbacks
        if not callbacks is None:
            for args in callbacks:
                self.callbacks.add(args)

        self.__dict__['loss'] = loss[0] if isinstance(loss, list) else loss
        self.__dict__['metrics'] = metrics
        self.__dict__['epochs'] = epochs
        self.__dict__['batch_size'] = batch_size
    def toJson(self):
        return {key: value if not (isinstance(value, Parameter) or isinstance(value, _ParameterSet))
                else value.toJson() for key, value in self}

    @classmethod
    def fromfile(cls, filepath):
        params = load(filepath + '/model_info.jbl')
        return cls(*params)

    def __getitem__(self, param):
        return self.__dict__[param]

    def __getattr__(self, param):
        return self.__dict__[param]

    def __iter__(self):
        return self.next()

    def _storeParamObj(self, param, param_type, param_key, error_str):
        raise NotImplementedError

    def next(self):
        for param in self.__dict__:
            yield (param, self.__dict__[param])
        raise StopIteration

    def getParamPath(self):
        path = self.prefix + '/' + self.optimizer.getPathStr() + '/'
        for layer in self.layers:
            if layer.identifier == 'Activation':
                path = path + layer.parameters['activation'] + '_'
            else:
                path = path + layer.identifier + '_'
        path = path[:-1] + '/'  # remove last '_'
        for layer in self.layers:
            path = path + layer.getPathStr()
        path = path[:-1] + '/' + self.loss
        for metric in self.metrics:
            str_metric = metric if isinstance(metric, str) else metric.__name__
            path = path + '_' + str_metric
        path = path + '_' + str(self.epochs)
        path = path + '_' + str(self.batch_size)

        import hashlib

        hash = hashlib.sha512(path).hexdigest()

        #return path
        return hash + '_%i' % self.input_shape[0]

    def toNpArray(self):
        #print type(self.callbacks)
        if self.callbacks is not None:
            cbks = self.callbacks.toNpArray() # Compatibility purposes
        else:
            cbks = None

        return [self.prefix,
                self.optimizer.toNpArray(),
                self.layers.toNpArray(),
                self.loss,
                self.metrics,
                self.epochs,
                self.batch_size,
                cbks,
#                self.scale,
                self.layers.toNpArray()[0][1]['input_shape']
                ]

class Parameter(object):
    def __init__(self, identifier, kwargs):
        self.__dict__ = dict()

        self.__name__ = identifier

        # DEPRECATED identifier, use __name__
        self.__dict__['identifier'] = identifier
        self.__dict__['parameters'] = kwargs

    def __getitem__(self, param):
        return self.__dict__[param]

    def __getattr__(self, param):
        return self.__dict__[param]

    def __iter__(self):
        return self.next()

    def next(self):
        for param in self.__dict__:
            yield (param, self.__dict__[param])
        raise StopIteration

    # def __eq__(self, param2):
    #     if self.__name__ == param2.__name__:
    #         for (arg1, value1), (arg2, value2) in zip(self.parameters, param2.parameters):
    #             if not arg1 == arg2 or not value1 == value2:
    #                 return False
    #         return True
    #     return False

    def toJson(self):
        return {'name': self.__name__,
                'args': self.parameters}

    def toNpArray(self):
        return [self.__name__, self.parameters]

    def _toKerasFn(self, keras_module, name, args):
        fn = getattr(keras_module, name)
        return fn(**args)


class _ParameterSet(object):
    def __init__(self):
        self.elements = list()

    # def add(self, element):
    #     if not isinstance(element, Parameter):
    #         raise ValueError('element type must be an instance of Parameter'
    #                          '%s of type %s was passed' % (element, type(element)))
    #     self.elements.append(element)

    def _add_many(self, instance_type, elements):
        # DEPRECATED
        if isinstance(elements, list):
            for item in elements:
                self._add_element(instance_type, item)
        #
        # for value in elements:
        #     self._add_element(instance_type, value)

    def _add_element(self, instance_type, params):
        self.elements.append(instance_type(params))

    def _add(self, instance_type, *args):

        if len(args) == 1:
            if isinstance(args[0], instance_type):
                self.elements.append(args[0])
            elif isinstance(args[0], dict):
                self._add_element(instance_type, args[0])
            elif isinstance(args[0], list): # DEPRECATED # TODO remove
                self._add_many(instance_type, args[0])
            else:
                raise NotImplementedError
                # raise ValueError('element must be a instance of Callback'
                #                  '%s of type %s was passed' % (element, type(element)))
        elif len(args) == 2:
            parameters = None
            name = None
            for arg in args:
                if isinstance(arg, dict):
                    parameters = arg
                elif isinstance(arg, str):
                    name = arg

            if name is None or parameters is None:
                raise NotImplementedError
            self._add_element(instance_type, parameters)
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        return self.elements[i]

    def __iter__(self):
        return self.next()

    def __len__(self):
        return len(self.elements)

    def next(self):
        for element in self.elements:
            yield element
        raise StopIteration

    def toJson(self):
        return {i:element.toJson() for i,element in enumerate(self.elements)}

    def pprint(self):
        raise NotImplementedError

    def toNpArray(self):
        return [element.toNpArray() for element in self.elements]

    def toKerasFn(self):
        return [element.toKerasFn() for element in self.elements]

    def fitMask(self, mask):
        if not isinstance(mask, _ParameterSet):
            raise ValueError('mask type must be an instance of _ParameterSet'
                             '%s of type %s was passed' % (mask, type(mask)))
        if not mask is None:
            for submask, element in zip(mask, self):
                if not element.fitMask(submask):
                    return False
        return True


class Optimizer(Parameter):
    method_list = ['SGD', 'Adam', 'Adagrad']

    def __init__(self, kwargs):
        method = kwargs["type"]
        del kwargs["type"]

        if not method in self.method_list:
            warn('Selected optmization method not found in method list.')

        super(Optimizer, self).__init__(method, kwargs)

    def toKerasFn(self):
        return self._toKerasFn(keras.optimizers, self.__name__, self.parameters)

    def getPathStr(self):
        return self.identifier


class Layer(Parameter):
    type_list = ['Conv2D', 'Conv1D', 'MaxPooling2D', 'MaxPooling1D', 'AveragePooling1D', 'AveragePooling2D', 'Flatten', 'Dense', 'Activation', 'Dropout']
    type2path = ['c2', 'c1', 'mp2', 'mp1', 'ap1', 'ap2', '', 'd', '', 'drop']

    def __init__(self, kwargs):
        layer_type = kwargs["type"]
        del kwargs["type"]

        if not layer_type in self.type_list:
            warn('Selected layer type not found in layer types list')

        super(Layer, self).__init__(layer_type, kwargs)

    def toKerasFn(self):
        return self._toKerasFn(keras.layers, self.__name__, self.parameters)

    def getPathStr(self):
        l_map = dict()
        for ident, abv in zip(self.type_list, self.type2path):
            l_map[ident] = abv

        param_str = l_map[self.identifier] + '_'
        for parameter in self.parameters:
            if isinstance(self.parameters[parameter], list) or isinstance(self.parameters[parameter], tuple):
                param_str = param_str + parameter[0] + '_'
                for element in self.parameters[parameter]:
                    param_str = param_str + str(element) + '_'
                param_str = param_str #+ '_'
            else:
                param_str = param_str + parameter[0] + '_' + str(self.parameters[parameter]) + '_'
        return param_str


class Callback(Parameter):
    type_list = ['ModelCheckpoint', 'EarlyStopping', 'ReduceLROnPlateau', 'CSVLogger']

    def __init__(self, kwargs):
        callback = kwargs["type"]
        del kwargs["type"]

        if not callback in self.type_list:
            warn('Selected callback not found in callbacks list')

        super(Callback, self).__init__(callback, kwargs)

    def toKerasFn(self):
        return super(Callback, self)._toKerasFn(keras.callbacks, self.identifier, self.parameters)


class Layers(_ParameterSet):
    def __init__(self, layers=None):
        super(Layers, self).__init__()

        if not layers is None:
            if isinstance(layers, list):
                for layer in layers:
                    if isinstance(layer, list):
                        new_layer = layer[1]
                        new_layer["type"] = layer[0]
                        layer = Layer(new_layer)
                    elif layer == None:
                        layer = Layer()
                    self.add(layer)
            else:
                raise ValueError('layers must be a instance of list'
                                 '%s of type %s was passed' % (layers, type(layers)))

    def add(self, *args):
        self._add(Layer, *args)


class Callbacks(_ParameterSet):
    def __init__(self, callbacks=None):
        super(Callbacks, self).__init__()

        if not callbacks is None:
            if isinstance(callbacks, list):
                for callback in callbacks:
                    if isinstance(callback, list):
                        callback = Callback(callback[0], **callback[1])
                    elif callback == None:
                        callback = Callback()
                    self.add(callback)
            else:
                raise ValueError('callbacks must be a instance of list'
                                 '%s of type %s was passed' % (callbacks, type(callbacks)))

    def add(self, *args):
        self._add(Callback, *args)


class _Path(object):
    MODEL_INFO_FILE_NAME = 'model_info.jbl'
    MODEL_STATE_FILE_NAME = 'model_state.h5'

    def __init__(self, home):
        self.subfolders = dict()
        self.subfolders['home'] = home
        pass

    def add(self, subfolder):
        self.subfolders[subfolder] = self.home + '/' + subfolder

    def createFolder(self):
        """Generate folders accordingly with the object structure"""
        raise NotImplementedError


class ConvolutionPaths(_Path):
    """Class with references to all subfolders(models and results) required for the convolutional models training and analysis"""
    results_path = './Analysis'

    def __init__(self):

        self.model_paths = self.getModelPaths(self.results_path)

    def getModelPaths(self, init_path):
        path_list = os.listdir(init_path)

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


# class ModelPaths(ConvolutionPaths):
#     def __init__(self, trnParams):
#         if not isinstance(trnParams, TrnParamsConvolutional):
#             raise ValueError('The parameters inserted must be an instance of TrnPararamsConvolutional'
#                              '%s of type %s was passed.'
#                              % (trnParams, type(trnParams)))
#
#         self.model_path = self.results_path + '/' + self.genPathStr(trnParams)
#         # self.model_data = "%s_%s_%s_%s"%(now.year,now.month,now.day,now.hour)
#
#         self.model_info_file = self.model_path + '/' + 'model_info.jbl'
#         self.model_info_file_txt = self.model_path + '/' + 'model_info.txt'
#         self.model_file = None
#         self.model_history = None
#         self.model_predictions = None
#         self.model_recovery_state = None
#         self.model_recovery_folds = None
#
#     def genPathStr(self, trnParams):
#         warn('Test path implemented')
#
#         return trnParams.getParamPath()
#
#     def selectFoldConfig(self, n_folds):
#         warn('Implement error handling')
#
#         self.fold_path = self.model_path + '/' + '%i_folds' % n_folds
#
#         self.model_file = self.fold_path + '/' + 'model_state.h5'
#         self.model_history = self.fold_path + '/' + 'history.csv'
#         self.model_predictions = self.fold_path + '/' + 'predictions'
#         self.model_recovery_state = self.fold_path + '/' + '~rec_state.h5'
#         self.model_recovery_folds = self.fold_path + '/' + '~rec_folds.jbl'
#
#         if exists(self.model_file):
#             return 'Trained'
#         # elif exists(self.model_recovery_state):
#         #   return 'Recovery'
#         else:
#             self.createFolders(self.model_path)
#             return 'Untrained'
#
#     @staticmethod
#     def createFolders(*args):
#         """Creates model folder structure from hyperparameters information"""
#         for folder in args:
#             print folder
#             mkdir(folder)
#
#     def purgeFolder(self, subfolder=''):
#         raise NotImplementedError

from sklearn.model_selection import ParameterGrid

class CmpndParameterGrid(ParameterGrid):
    def __init__(self, param_grid, compound_params=None):
        if compound_params is None:
            compound_params = ['optimizer', 'layers']

        for key in param_grid.keys():
            if str(key).lower() in compound_params:
                param_grid[key] = [params for params in ParameterGrid(param_grid[key])]
        super(CmpndParameterGrid, self).__init__(param_grid)
