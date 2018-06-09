""" 
  This file contents all log functions
"""
import contextlib
import wave

import keras
import numpy as np
import numpy.random as np_rnd
import os

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.utils import shuffle

from Functions.SystemIO import filterPaths, listfolders


class DataHandlerFunctions(object):
    def __init__(self):
        self.name = 'DataHandler Class'

    def CreateEventsForClass(self, data, n_events, method='reply'):
        print '%s: CreateEventsForClass' % (self.name)
        print 'Original Size: (%i, %i)' % (data.shape[0], data.shape[1])
        if n_events == 0:
            return data
        else:
            if method == 'reply':
                appended_data = data[np_rnd.random_integers(0, data.shape[0] - 1, size=n_events), :]
                return_data = np.append(data, appended_data, axis=0)
                return return_data


def trgt2categorical(trgt, n_classes):
    return keras.utils.to_categorical(trgt, num_classes=n_classes)


def lofar2image(all_data, all_trgt, class_labels, class_window, stride, verbose=False, filepath='./lofar_images', dtype = np.float64):

    data_classes = [all_data[all_trgt == class_type] for class_type in class_labels]

    data_shape = (sum([category.shape[0] - class_window for category in data_classes]),
                  class_window,
                  all_data.shape[1])

    image_data = np.memmap(filename = filepath, shape= data_shape, mode = 'w+', dtype = dtype)
    trgt_image = np.zeros(shape=data_shape[0])

    for class_type in class_labels:
        events = all_data[all_trgt == class_type]

        if verbose:
            print "Class %s:" % class_type
            print "   Qt Samples: %s" % events.shape[0]
            print "   Freq. Bins: %s" % events.shape[1]

        for event_index in range(0, events.shape[0], stride):
            if not event_index > events.shape[0] - class_window:
                image_data[event_index] = events[event_index:event_index + class_window, :]
                trgt_image[event_index] = class_type
    if verbose:
        print "\nProcessed dataset shape:"
        print "     Qt Samples:     %s" % image_data.shape[0]
        print "     Samples length: %s" % image_data.shape[1]
        print "     Samples width:  %s" % image_data.shape[2]

    return [image_data, trgt_image]


def Kfold(dataset, k, shuffle=False, stratify=False):
    """
    Envelop function for folding operation
    """
    # remove class labels
    data = dataset[0]
    if stratify:
        kf = StratifiedKFold(k, shuffle)
        return kf.split(dataset[0], dataset[1])

    kf = KFold(k, shuffle)
    return kf.split(data)


class SonarRunsKFold(BaseCrossValidator):
    INPUTDATAPATH = '/home/pedrolisboa/Workspace/lps/Marinha/Data/SONAR/Classification/4classes'

    def __init__(self,
                 validation_runs = {'ClassA': 1,
                                    'ClassB': 2,
                                    'ClassC': 2,
                                    'ClassD': 2},
                 shuffle=True):
        super(SonarRunsKFold, self).__init__()
        self.shuffle = shuffle
        self.validation_runs = validation_runs

    def _iter_test_indices(self, X=None, y=None, groups=None, dev = False):
        run_dict = dict()
        class_offset = 0
        for class_folder in listfolders(self.INPUTDATAPATH):
            run_files = listfolders(self.INPUTDATAPATH + '/' + class_folder)
            run_paths = map(lambda x: self.INPUTDATAPATH + '/' + class_folder + '/' + x, run_files)
            run_indexes = list(self._iterClassIndices(run_paths, class_offset, 1024))

            if dev:
                offsets = list(map(lambda x: x[0], run_indexes))
                lengths = list(map(len, run_indexes))
                print class_folder
                print "\tLength\tOffset"
                for (i, length), offset in zip(enumerate(lengths), offsets):
                    print "Run %i:\t%i\t%i" % (i,length, offset)
                print "Total: \t%i\n" % (sum(lengths))

            class_offset = class_offset + sum(map(len, run_indexes))

            if self.shuffle:
                run_indexes = shuffle(run_indexes)

            run_dict[class_folder] = run_indexes

        # get class with minimum number of runs
        ref_class_i = np.argmin(map(len, run_dict.values()))
        n_folds = len(run_dict.values()[ref_class_i]) # number of folds will be limited by
                                                     # the class with minimum number of runs
        if dev:
            print "Total: %i" % n_folds

        for fold in range(n_folds):
            #DEBUG
            if dev:
                print "Fold %i" % fold
                val_indices = np.array([], dtype=np.int)

            test_indices = np.array([], dtype=np.int)
            for cls_key in run_dict:
                val_runs = self.validation_runs[cls_key]  # number of runs for validation set
                class_runs = run_dict[cls_key]

                #DEBUG
                if dev:
                    print "\t%s: Inf: %i \t Sup: %i" % (cls_key, fold*val_runs, val_runs*(fold + 1))
                    val_check = np.concatenate(class_runs[fold*val_runs:val_runs*(fold + 1)])
                    val_indices = np.concatenate((val_indices, val_check))

                test_runs = class_runs[:fold*val_runs] + class_runs[val_runs*(fold + 1):] # cutout val runs
                test_runs = np.concatenate(test_runs)
                test_indices = np.concatenate((test_indices,test_runs))

            if dev:
                total = test_indices.shape[0] + val_indices.shape[0]
                test_per, val_per = 100*test_indices.shape[0]/total, 100*val_indices.shape[0]/total
                print "\n\tSamples -> Train: %i (%i%%)  Val: %i (%i%%)  Total: %i\n" % \
                      (test_indices.shape[0], test_per, val_indices.shape[0], val_per, total)

            yield test_indices

        if dev:
            yield run_dict

    def _iter_test_indices_old(self, X=None, y=None, groups=None, dev = False):
        run_dict = dict()
        class_offset = 0
        for class_folder in listfolders(self.INPUTDATAPATH):
            run_files = listfolders(self.INPUTDATAPATH + '/' + class_folder)
            run_paths = map(lambda x: self.INPUTDATAPATH + '/' + class_folder + '/' + x, run_files)
            run_indexes = list(self._iterClassIndices(run_paths, class_offset, 1024))

            if dev:
                offsets = list(map(lambda x: x[0], run_indexes))
                lengths = list(map(len, run_indexes))
                print class_folder
                print "\tLength\tOffset"
                for (i, length), offset in zip(enumerate(lengths), offsets):
                    print "Run %i:\t%i\t%i" % (i,length, offset)
                print "Total: \t%i\n" % (sum(lengths))

            class_offset = class_offset + sum(map(len, run_indexes))

            run_dict[class_folder] = run_indexes
        return run_dict

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

    def get_n_splits(self, X=None, y=None, groups=None):
        raise NotImplementedError