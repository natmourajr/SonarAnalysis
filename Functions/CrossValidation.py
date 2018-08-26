""" 
  Author: Pedro Henrique Braga Lisboa
          pedrolisboa at poli.ufrj.br
  This module contains all Cross-Validation utilities
"""
import contextlib
import re
import wave
from collections import OrderedDict
from itertools import cycle, islice

import numpy as np
import os

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.utils import shuffle

from Functions import SystemIO
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
            run_names.sort()
            run_paths.sort()
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