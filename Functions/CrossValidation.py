"""
  Author: Pedro Henrique Braga Lisboa
          pedrolisboa at poli.ufrj.br
  This module contains all Cross-Validation utilities
"""
import contextlib
import re
import wave
from collections import OrderedDict
from itertools import cycle, islice, product, combinations

import numpy as np
import os

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.utils import shuffle

from Functions import SystemIO
from Functions.NpUtils.DataTransformation import SonarRunsInfo
from Functions.SystemIO import listfiles, load, save, listfolders


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
    def __init__(self, n_cvs, n_folds,cvs_paths, inputdatapath, db_name='4classes', cv_filemask='10_folds_cv_runs_nested_'):
        self.resultspath = cvs_paths
        self.audiodatapath = inputdatapath + '/' + db_name
        self.datapath = cvs_paths + '/' + db_name
        self.n_cvs = n_cvs
        self.n_folds = n_folds
        self.cv_filemask = cv_filemask
        self.info = SonarRunsInfo(self.audiodatapath)

    def __iter__(self):
        return self.next()

    def next(self):
        for cv_name, cv in self.cv.items():
            yield cv_name, cv
        raise StopIteration

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


class SonarRunsCV(BaseCrossValidator):
    def __init__(self, n_splits, inputdatapath, window, overlap=0, decimation_rate=1, verbose=False):
        super(SonarRunsCV, self).__init__()

        runs_info = SonarRunsInfo(inputdatapath, window, overlap, decimation_rate, verbose)
        self.inputdatapath = inputdatapath
        self.n_splits = n_splits
        self.class_folders = runs_info.class_folders
        self.runs = runs_info.runs
        self.runs_named = runs_info.runs_named

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

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


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

        for class_folder in listfolders(self.INPUTDATAPATH):
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
