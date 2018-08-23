""" 
  This file contents all log functions
"""
import contextlib
import re
import wave
from collections import OrderedDict
from itertools import combinations, product, cycle, islice

import keras
import numpy as np
import numpy.random as np_rnd
import os

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.utils import shuffle

from Functions import SystemIO
from Functions.SystemIO import filterPaths, listfiles, load, save, listfolders


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


# def lofar2image(all_data, all_trgt, class_labels, class_window, stride, verbose=False, filepath='./lofar_images', dtype = np.float64):
#
#     data_classes = [all_data[all_trgt == class_type] for class_type in class_labels]
#
#     data_shape = (sum([category.shape[0] - class_window for category in data_classes]),
#                   class_window,
#                   all_data.shape[1])
#
#     image_data = np.memmap(filename = filepath, shape= data_shape, mode = 'w+', dtype = dtype)
#     trgt_image = np.zeros(shape=data_shape[0])
#
#     for class_type in class_labels:
#         events = all_data[all_trgt == class_type]
#
#         if verbose:
#             print "Class %s:" % class_type
#             print "   Qt Samples: %s" % events.shape[0]
#             print "   Freq. Bins: %s" % events.shape[1]
#
#         for event_index in range(0, events.shape[0], stride):
#             if not event_index > events.shape[0] - class_window:
#                 image_data[event_index] = events[event_index:event_index + class_window, :]
#                 trgt_image[event_index] = class_type
#     if verbose:
#         print "\nProcessed dataset shape:"
#         print "     Qt Samples:     %s" % image_data.shape[0]
#         print "     Samples length: %s" % image_data.shape[1]
#         print "     Samples width:  %s" % image_data.shape[2]
#
#     return [image_data, trgt_image]

def lofar2image(all_data, all_trgt, index_info, class_labels, class_window, stride, verbose=False, run_split_info=None,
                debug = False,
                filepath=None,
                dtype=np.float64):
    if run_split_info is not None:
        fold_runs = np.concatenate([np.extract([np.isin(run, index_info).all() for run in cls_runs], cls_runs)
                                    for cls_runs in run_split_info.runs.values()])
        pruned_indexes = np.concatenate([range(run[0], run[-1] - class_window, stride) for run in fold_runs])
    else:
        # Extract the indexes of elements of a given eventclass
        i_class_extract = lambda class_type: np.where(all_trgt == class_type)[0]
        # Generate an iterator for sample extraction
        pruned_iterator = lambda class_type: range(i_class_extract(class_type)[0],
                                                   i_class_extract(class_type)[-1] - class_window,
                                                   stride)
        pruned_indexes = np.concatenate(map(lambda class_key: pruned_iterator(class_key), class_labels))

    data_shape = (pruned_indexes.shape[0],
                  class_window,
                  all_data.shape[1],
                  1)
    if not filepath is None:
        image_data = np.memmap(filename=filepath, shape=data_shape, mode='w+', dtype=dtype)
    else:
        image_data = np.zeros(shape=data_shape)

    trgt_image = np.zeros(shape=data_shape[0])

    for image_index, spectre_index in enumerate(pruned_indexes):
        new_data = all_data[spectre_index:spectre_index + class_window, :]
        new_data = np.array(new_data.reshape(new_data.shape[0], new_data.shape[1], 1), np.float64)
        image_data[image_index] = new_data
        trgt_image[image_index] = all_trgt[spectre_index]

    if verbose:
        print "Image dataset shape:"
        for class_key in class_labels:
            print "     %s (%i) samples: %i" % (class_labels[class_key], class_key, len(pruned_iterator(class_key)))
        print ""
        print "     Samples Total:   %s" % image_data.shape[0]
        print "     Sample length:   %s" % image_data.shape[1]
        print "     Sample width:    %s" % image_data.shape[2]
        print ""
        if image_data.nbytes < 10 ** 6:
            print "     File size:   Kb"
        elif image_data.nbytes < 10 ** 9:
            print "     File size:   %i Mb" % int(image_data.nbytes / 10 ** 6)
        else:
            print "     File size:   Gb"

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

class NestedCV:
    def __init__(self, n_cvs, n_folds,cvs_paths, audiodatapath, db_name='4classes', cv_filemask='10_folds_cv_runs_nested_'):
        self.resultspath = cvs_paths
        self.audiodatapath = audiodatapath
        self.datapath = cvs_paths + '/' + db_name
        self.n_cvs = n_cvs
        self.n_folds = n_folds
        self.cv_filemask = cv_filemask

    def createCVs(self, data, trgt):
        self.cv = {self.cv_filemask + '%i' % cv_i: list(SonarRunsCV(self.n_folds, self.audiodatapath).split(data,trgt))
                   for cv_i in range(self.n_cvs)}
        for cv_filename, cv_configuration in self.cv.items():
            save(cv_configuration, self.resultspath + '/' + cv_filename)

    def loadCVs(self):
        def isFoldConfig(x):
            return not re.search(self.cv_filemask, x) is None

        self.cv = {cv_filename: load(self.resultspath + '/' + cv_filename)
                   for cv_filename in filter(isFoldConfig, os.listdir(self.resultspath))}

    def exists(self):
        checkedfiles = [SystemIO.exists(self.resultspath + '/' + self.cv_filemask + '%i' % cv_i)
                        for cv_i in range(self.n_cvs)]
        return not False in checkedfiles

class sSonarRunsKFold(BaseCrossValidator):
    def __init__(self, n_splits, inputdatapath, shuffle = False, dev = False):
        super(sSonarRunsKFold, self).__init__()
        self.inputdatapath = inputdatapath
        self.dev = dev
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.runs = OrderedDict()

        # Load runs from folders
        class_offset = 0
        self.class_folders = list(listfolders(self.inputdatapath))
        for class_folder in listfolders(self.inputdatapath):
            run_files = listfiles(self.inputdatapath + '/' + class_folder + '/')
            run_paths = map(lambda x: self.inputdatapath + '/' + class_folder + '/' + x, run_files)
            run_indices = list(self._iterClassIndices(run_paths, class_offset, 1024))


            if self.dev:
                offsets = list(map(lambda x: x[0], run_indices))
                lengths = list(map(len, run_indices))
                print class_folder
                print "\tLength\tOffset"
                for (i, length), offset in zip(enumerate(lengths), offsets):
                    print "Run %i:\t%i\t%i" % (i, length, offset)
                print "Total: \t%i\n" % (sum(lengths))

            class_offset = class_offset + sum(map(len, run_indices))

            self.runs[class_folder] = run_indices

    def _iter_test_indices(self, X=None, y=None, groups=None, dev=False):
        dataset_perms = dict()
        for class_folder in listfolders(self.inputdatapath):
            run_perm = np.array([range(0, self.n_splits - 1)
                                 for _ in self.runs[class_folder]])

            shuffle_perm = np.apply_along_axis(shuffle, axis=1, arr=run_perm)
            dataset_perms[class_folder] = shuffle_perm

        for class_folder, shuffle_perm in dataset_perms.items():
            print shuffle_perm.shape

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

class SonarRunsCV(BaseCrossValidator):
    def __init__(self, n_splits, inputdatapath, verbose = False):
        super(SonarRunsCV, self).__init__()
        self.inputdatapath = inputdatapath
        self.n_splits = n_splits
        self.runs = OrderedDict()
        self.runs_named = OrderedDict()
        self.verbose = verbose

        # Load runs from folders
        class_offset = 0
        self.class_folders = list(listfolders(self.inputdatapath))
        for class_folder in self.class_folders:
            run_files = listfiles(self.inputdatapath + '/' + class_folder)
            run_files = list(run_files)
            run_names = map(lambda x: str(x[:-4]), run_files) # remove .wav file extension
            run_paths = map(lambda x: self.inputdatapath + '/' + class_folder + '/' + x, run_files)
            run_indices = list(self._iterClassIndices(run_paths, class_offset, 1024))

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

    def _iter_test_indices(self, X=None, y=None, groups=None, dev=False):
        run_cyclers = {cls: cycle(shuffle(self.runs[cls])) for cls in self.class_folders}
        def getClsRuns(class_folders):
            # test_indices = [index for cls in self.class_folders
            #                 for run in islice(run_cyclers[cls],
            #                                   int(round(len(self.runs[cls]) / self.n_splits)))
            #                 for index in run]

            for cls in class_folders:
                n_runs = len(self.runs[cls])
                qnt_runs = int(round(float(n_runs) / self.n_splits))
                for run in islice(run_cyclers[cls], qnt_runs):
                    for index  in run:
                        yield index
        for _ in range(self.n_splits):
            test_indices = np.fromiter(getClsRuns(self.class_folders), dtype=int)
            yield test_indices


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


class SonarRunsKFold(BaseCrossValidator):
    INPUTDATAPATH = '/home/pedrolisboa/Workspace/lps/Marinha/Data/SONAR/Classification/4classes'

    def __init__(self,
                 n_splits,
                 shuffle=True,
                 validation_runs=None,
                 dev=False,
                 split_A=False,
                 split_All=None,
                 val_share=2):
        super(SonarRunsKFold, self).__init__()
        self.val_share = val_share
        self.dev = dev
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.runs = OrderedDict()

        # Load runs from folders
        class_offset = 0
        for class_folder in listfiles(self.INPUTDATAPATH):
            run_files = listfiles(self.INPUTDATAPATH + '/' + class_folder)
            run_paths = map(lambda x: self.INPUTDATAPATH + '/' + class_folder + '/' + x, run_files)
            run_indices = list(self._iterClassIndices(run_paths, class_offset, 1024))

            if split_A and class_folder == 'ClassA':
                new_run_indices = list()
                for indices in run_indices:
                    new_run_indices.append(indices[:(len(indices) / 2)])
                    new_run_indices.append(indices[(len(indices) / 2):])
                run_indices = new_run_indices

            if not split_All is None:
                for cls in split_All:
                    if cls == class_folder:
                        new_run_indices = list()
                        for indices in run_indices:
                            for i in range(0, split_All[cls]):
                                if i == split_All[cls] - 1:
                                    new_run_indices.append(indices[i * (len(indices) / split_All[cls]):])
                                else:
                                    new_run_indices.append(indices[i * (len(indices) / split_All[cls]):(i + 1) * (
                                            len(indices) / split_All[cls])])
                        run_indices = new_run_indices

            if self.dev:
                offsets = list(map(lambda x: x[0], run_indices))
                lengths = list(map(len, run_indices))
                print class_folder
                print "\tLength\tOffset"
                for (i, length), offset in zip(enumerate(lengths), offsets):
                    print "Run %i:\t%i\t%i" % (i, length, offset)
                print "Total: \t%i\n" % (sum(lengths))

            class_offset = class_offset + sum(map(len, run_indices))

            self.runs[class_folder] = run_indices

        if validation_runs is None:
            # To implement: search class folders and set validation_runs based on the number of total samples
            validation_runs = {}
            cls_min_runs = np.amin(map(len, self.runs.values()))  # get class with minimum number of runs
            if self.dev:
                print "N of Runs for test fold:"
            for class_folder in self.runs:
                validation_runs[class_folder] = int(round(len(self.runs[class_folder]) / float(cls_min_runs)))
                if self.dev:
                    print "%s -> %i" % (class_folder, validation_runs[class_folder])
        self.validation_runs = validation_runs

    def _iter_test_indices(self, X=None, y=None, groups=None, dev=False):
        run_combs = dict()
        for cls, value in self.runs.items():
            run_combs[cls] = list(combinations(value, self.validation_runs[cls]))
        fold_configs = list(product(*run_combs.values()))
        if self.shuffle:
            fold_configs = shuffle(fold_configs)
        fold_configs = cycle(fold_configs)
        for _ in range(self.n_splits):
            test_indices = list()
            for i in range(0, self.val_share):
                test_indices.append(np.concatenate([run for cls_runs in fold_configs.next() for run in cls_runs]))
            test_indices = np.array(test_indices)

            test_indices = np.concatenate(test_indices)
            print test_indices.shape

            if dev:
                print "\n\tTest Fold Size-> %i (%f%%)" % (test_indices.shape[0], float(test_indices.shape[0]) / 77561)

            yield test_indices

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
