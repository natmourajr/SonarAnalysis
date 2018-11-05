"""
This module contains utilities to handle and transform lofar data
"""
import contextlib
import gc
import os
import warnings
import wave
from collections import OrderedDict

import keras
import numpy as np
from keras.utils import to_categorical
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_memory

from Functions.SystemIO import listfolders, listfiles, load, exists, save


def trgt2categorical(trgt, n_classes):
    return keras.utils.to_categorical(trgt, num_classes=n_classes)


class SonarRunsInfo():
    def __init__(self, inputdatapath, window, verbose=False):
        self.inputdatapath = inputdatapath
        self.runs = OrderedDict()
        self.runs_named = OrderedDict()
        self.verbose = verbose

        # Load runs from folders
        class_offset = 0
        self.class_folders = list(listfolders(self.inputdatapath))
        self.class_folders.sort()
        for class_folder in self.class_folders:
            run_files = listfiles(self.inputdatapath + '/' + class_folder)
            run_files = list(run_files)
            run_names = map(lambda x: str(x[:-4]), run_files) # remove .wav file extension
            run_paths = map(lambda x: self.inputdatapath + '/' + class_folder + '/' + x, run_files)
            run_names.sort()
            run_paths.sort()
            run_indices = list(self._iterClassIndices(run_paths, class_offset, window))
            if self.verbose:
                offsets = list(map(lambda x: x[0], run_indices))
                lengths = list(map(len, run_indices))
                print class_folder
                print "\tLength\tOffset"
                for (i, length), offset in zip(enumerate(lengths), offsets):
                    print "Run %i:\t%i\t%i" % (i, length, offset)
                print "Total: \t%i\n" % (sum(lengths))

            class_offset = class_offset + sum(map(len, run_indices))

            self.runs[class_folder] = run_indices
            self.runs_named[class_folder] = {filename: indices for filename, indices in zip(run_names, run_indices)}

    def _iterClassIndices(self, runpaths, class_offset, window):
        run_offset = 0
        for run in runpaths:
            run_indices = self._getRunIndices(run, class_offset + run_offset, window)
            run_offset += len(run_indices)
            yield run_indices

    def _getRunIndices(self, runfile, offset, window):
        with contextlib.closing(wave.open(runfile, 'r')) as runfile:
            frames = runfile.getnframes()
            end_frame = frames / window + offset

        return range(offset, end_frame)

class Scaler:
    def __init__(self, mode='std'):
        self.mode = mode

    def fit(self, X, y):
        pass

    def transform(self, X, y):
        pass


class Lofar2Image(BaseEstimator, TransformerMixin):
    def __init__(self, all_data, all_trgt, window_size, stride, run_indices_info,
                 filepath = None, channel_axis='last', dtype=np.float64, verbose=0, memory=None):
        super(Lofar2Image, self).__init__()
        self.window_size = window_size
        self.run_indices_info = run_indices_info
        self.filepath = filepath
        self.dtype = dtype
        self.stride = stride
        self.all_data = all_data
        self.all_trgt = all_trgt
        self.verbose = verbose

        self.memory = memory
        self.pruned_indexes = None
        self.data_shape = None

        if channel_axis is None:
            channel_axis = 'last'
        if not channel_axis.lower() in ['first', 'last', 'none']:
            warnings.warn('Desired channel dimension is invalid, '
                          'fallback to last dimension.',
                           RuntimeWarning)
            channel_axis = 'last'
        self.channel_dim = channel_axis.lower()

    def set_params(self, **params):
        for key, value in params.items():
            print key
            print value
            setattr(self, key, value)

        return self

    def extractRuns(self, lofar_data, lofar_trgt, shattered=False):
        # class_labels = {'ClassA':0, 'ClassB':1, 'ClassC':2, 'ClassD':3}
        if shattered:
            raise NotImplementedError
        else:
            for cls_runs in self.run_indices_info.runs_named.values():
                for run_name, run in cls_runs.items():
                    run_data, run_trgt = self.all_data[run], self.all_trgt[run]
                    cls_i = np.unique(run_trgt)[0]
                    # lofar = lofar_data[lofar_trgt == cls_i]
                    # print cls_i
                    # print np.isin(run_data, lofar, assume_unique=True).any(axis=1)
                    # print run_data.shape
                    # print lofar.shape
                    if np.isin(run_data, lofar_data, assume_unique=False).all(axis=1).all():
                        yield (cls_i, run_name), run_data

    # def _getClsFromRunName(self, run_name):
    #     return int(run_name[-2])

    # def _genRunTrgt(self, run_name, run_data):
    #     i_cls = self._getClsFromRunName(run_name)
    #     return i_cls*np.ones(run_data.shape[0])

    def _segmentRun(self, run, trgt, window, stride):
        image_run =  np.concatenate([run[np.newaxis, index:(index+window)]
                                    for index in range(0, run.shape[0] - window, stride)],
                                    axis=0)
        image_trgt = trgt * np.ones(image_run.shape[0])

        return image_run, image_trgt

    def _genImageDataset(self, run_data):
        runs = np.empty(len(run_data), dtype=np.ndarray)
        trgts = np.empty(len(run_data), dtype=np.ndarray)
        for i, (trgt, run) in enumerate(run_data):
            runs[i], trgts[i] = self._segmentRun(run, trgt, self.window_size, self.stride)

        return np.concatenate(runs, axis=0), \
               np.concatenate(trgts, axis=0)

    def fit(self, X, y):
        if y.ndim > 1: # recover labels from categorical vector
            self.sparse_format = True
            y = y.argmax(axis=1)
        else:
            self.sparse_format = False

        self.y = y

        return self

    def transform(self, X, y=None):
        memory = check_memory(self.memory)

        _transform_cached = memory.cache(self._transform)

        return _transform_cached(X, y)

    def _transform(self, X, y=None):
        if y is None:
            y = self.y
        # if X.shape[0] != y.shape[0]:
        #     print X.shape
        #     print y.shape
        #     raise ValueError
        separated_runs = OrderedDict({(run_trgt, run_name): run_data
                                      for (run_trgt, run_name), run_data in self.extractRuns(X, y)})

        if self.verbose:
            print "Runs found"
            for key in self.separated_runs.keys():
                print key

        image_X, image_y = self._genImageDataset([(trgt, run)
                                                 for (trgt, _), run in separated_runs.items()])

        # print image_X.shape
        # print image_y.shape

        print 'image shape'
        print image_X.shape
        print image_y.shape

        if self.channel_dim == 'first':
            image_X = image_X[np.newaxis, :, :, :]
        elif self.channel_dim == 'last':
            image_X = image_X[:, :, :, np.newaxis]

        #if self.sparse_format:
        image_y = to_categorical(image_y)

        gc.collect()

        return image_X, image_y
    #
    # def gen_new_cv(self, all_data, all_trgt, cv):
    #     offset = 0
    #     new_cv = list()
    #
    #     for i, (train, test) in enumerate(cv):
    #         new_data, new_trgt = self.transform(all_data[train], all_trgt[train])
    #
    #         new_test_data, new_test_trgt = self.transform(all_data[test], all_trgt[test])
    #
    #         new_train = np.arange(new_data.shape[0]) + offset
    #         offset += new_train[-1] + 1
    #         new_test = np.arange(new_test_data.shape[0]) + offset
    #         offset += new_test[-1] + 1
    #
    #         new_cv.append((new_train, new_test))
    #
    #     def gen_new_data(train, test):
    #         new_data, new_trgt = self.transform(all_data[train], all_trgt[train])
    #
    #         new_test_data, new_test_trgt = self.transform(all_data[test], all_trgt[test])
    #
    #         return np.vstack([new_data, new_test_data])
    #
    #     def gen_new_trgt(train, test):
    #         new_data, new_trgt = self.transform(all_data[train], all_trgt[train])
    #
    #         new_test_data, new_test_trgt = self.transform(all_data[test], all_trgt[test])
    #
    #         return np.vstack([new_trgt, new_test_trgt])
    #
    #     image_data = np.vstack([gen_new_data(train, test) for train, test in cv])
    #     image_trgt = np.vstack([gen_new_trgt(train, test) for train, test in cv])
    #
    #     return image_data, image_trgt, new_cv
    #
    #

def lofar2image(all_data, all_trgt,
                index_info, window_size, stride,
                run_indices_info,
                filepath=None,
                dtype=np.float64):

    fold_runs = np.concatenate([np.extract([np.isin(run, index_info).all() for run in cls_runs], cls_runs)
                                for cls_runs in run_indices_info.runs.values()])
    pruned_indexes = np.concatenate([range(run[0], run[-1] - window_size, stride) for run in fold_runs])

    data_shape = (pruned_indexes.shape[0],
                  window_size,
                  all_data.shape[1],
                  1)
    if not filepath is None:
        image_data = np.memmap(filename=filepath, shape=data_shape, mode='w+', dtype=dtype)
    else:
        image_data = np.zeros(shape=data_shape, dtype=dtype)

    trgt_image = np.zeros(shape=data_shape[0])

    for image_index, spectre_index in enumerate(pruned_indexes):
        new_data = all_data[spectre_index:spectre_index + window_size, :]
        new_data = np.array(new_data.reshape(new_data.shape[0], new_data.shape[1], 1), np.float64)
        image_data[image_index] = new_data
        trgt_image[image_index] = all_trgt[spectre_index]

    return [image_data, trgt_image]

def lofar_mean(data, trgt, sliding_window):
    def ndma(data, sliding_window, axis):
        return np.concatenate([np.mean(sample, axis=axis, keepdims=True)
                               for sample in np.split(data, data.shape[axis]/sliding_window, axis=axis)],
                               axis=axis)

    def average_samples(data, trgt, cls_i):
        cls_data = data[trgt == cls_i]
        cls_averaged_data = ndma(cls_data, sliding_window, axis=1)
        return cls_averaged_data

    averaged_data = np.concatenate([average_samples(data, trgt, cls_i) for cls_i in np.unique(trgt)],
                                   axis=0)
    return averaged_data


