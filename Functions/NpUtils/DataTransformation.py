"""
This module contains utilities to handle and transform lofar data
"""
import contextlib
import wave
from collections import OrderedDict

import keras
import numpy as np

from Functions.SystemIO import listfolders, listfiles


def trgt2categorical(trgt, n_classes):
    return keras.utils.to_categorical(trgt, num_classes=n_classes)


class SonarRunsInfo():
    def __init__(self, inputdatapath, verbose = False):
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
