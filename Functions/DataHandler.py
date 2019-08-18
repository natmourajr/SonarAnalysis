"""
  This file contents all log functions
"""
import os

import numpy as np
import numpy.random as np_rnd
from sklearn.externals import joblib


class LofarDataset(object):
    def __init__(self, data_path):
        self.data_path = data_path

    def loadData(self, database, n_pts_fft, overlap_fft, decimation_rate, spectrum_bins_left):
        data_path = self.data_path
        # Check if LofarData has been created
        if not os.path.exists('%s/%s/lofar_data_file_fft_overlap_%i_%i_decimation_%i_spectrum_left_%i.jbl' %
                              (data_path, database, n_pts_fft, overlap_fft, decimation_rate, spectrum_bins_left)):
            print 'No Files in %s/%s\n' % (data_path, database)
            return
        else:
            # Read lofar data
            [data, trgt, class_labels] = joblib.load('%s/%s/lofar_data_file_fft_overlap_%i_%i_decimation_%i_spectrum_left_%i.jbl' %
                                                     (data_path, database, n_pts_fft, overlap_fft, decimation_rate,
                                                      spectrum_bins_left))

            dataset = (data, trgt, class_labels)
        return dataset

class DataHandlerFunctions(object):
    def __init__(self):
        self.name = 'DataHandler Class'

    def CreateEventsForClass(self, data, n_events, method='reply'):
        print('{}: CreateEventsForClass'.format(self.name))
        print('Original Size: ({}, {})'.format(data.shape[0], data.shape[1]))
        if n_events == 0:
            return data
        else:
            if method == 'reply':
                appended_data = data[np_rnd.random_integers(0, data.shape[0] - 1, size=n_events), :]
                return_data = np.append(data, appended_data, axis=0)
                return return_data
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

class DataBalancer(DataHandlerFunctions):
    def __init__(self):
        super(DataBalancer, self).__init__()

    def getBiggestClass(self, trgt, class_labels, verbose = 0):
        qtd_events_biggest_class = 0
        biggest_class_label = ''

        for iclass, class_label in enumerate(class_labels):
            if sum(trgt == iclass) > qtd_events_biggest_class:
                qtd_events_biggest_class = sum(trgt == iclass)
                biggest_class_label = class_label
            if verbose:
                print "Qtd event of %s is %i" % (class_label, sum(trgt == iclass))
        if verbose:
            print "\nBiggest class is %s with %i events" % (biggest_class_label, qtd_events_biggest_class)

        return qtd_events_biggest_class, biggest_class_label

    def oversample(self, data, trgt, class_labels, development_flag=False, qtd_development_events=100, verbose=0):
        balanced_data = {}
        balanced_trgt = {}

        qtd_events_biggest_class, _ = self.getBiggestClass(trgt, class_labels, verbose=verbose)

        for iclass, class_label in enumerate(class_labels):
            if development_flag:
                class_events = data[trgt == iclass, :]
                if len(balanced_data) == 0:
                    balanced_data = class_events[0:qtd_development_events, :]
                    balanced_trgt = (iclass) * np.ones(qtd_development_events)
                else:
                    balanced_data = np.append(balanced_data,
                                              class_events[0:qtd_development_events, :],
                                              axis=0)
                    balanced_trgt = np.append(balanced_trgt, (iclass) * np.ones(qtd_development_events))
            else:
                if len(balanced_data) == 0:
                    class_events = data[trgt == iclass, :]
                    balanced_data = self.CreateEventsForClass(
                        class_events, qtd_events_biggest_class - (len(class_events)))
                    balanced_trgt = (iclass) * np.ones(qtd_events_biggest_class)
                else:
                    class_events = data[trgt == iclass, :]
                    created_events = (self.CreateEventsForClass(data[trgt == iclass, :],
                                                                qtd_events_biggest_class -
                                                                (len(class_events))))
                    balanced_data = np.append(balanced_data, created_events, axis=0)
                    balanced_trgt = np.append(balanced_trgt,
                                              (iclass) * np.ones(created_events.shape[0]), axis=0)
        return balanced_data, balanced_trgt
