""" Projeto Marinha do Brasil
    Autor: Pedro Henrique Braga Lisboa
    Laboratorio de Processamento de Sinais - UFRJ
    Laboratorio de Tecnologia Sonar - UFRJ/Marinha do Brasil
"""
import os
import numpy as np
import joblib
from .lofar import lofar, tpsw

class LofarAnalysis:
    CLS_MAPPING = {'ClassA': 0, 'ClassB': 1, 'ClassC': 2, 'ClassD': 3}

    def __init__(self, decimation_rate, n_pts_fft, n_overlap, spectrum_bins_left='auto', **tpswargs):
        self.decimation_rate = decimation_rate
        self.stftargs = {'n_pts_fft': n_pts_fft,
                         'n_overlap': n_overlap}
        self.tpswargs = tpswargs
        self.lofar_data = None
        self.spectrum_bins_left = spectrum_bins_left

    def from_raw_data(self, raw_data, fs_data, database, outputpath=None, verbose=1):
        #input_db_path = os.path.join(inputpath, database)
        output_db_path = os.path.join(outputpath, database)

        decimation_rate = self.decimation_rate
        n_pts_fft = self.stftargs['n_pts_fft']
        n_overlap = self.stftargs['n_overlap']
        spectrum_bins_left = self.spectrum_bins_left
        lofar_data = self._from_raw_data(raw_data, fs_data, verbose)

        if spectrum_bins_left is 'auto':
            spectrum_bins_left = lofar_data.shape[0]
        if outputpath is not None:
            joblib.dump(lofar_data, os.path.join(output_db_path,
                                                 'lofar_data_file_fft_%i_overlap_%i_decimation'
                                                 '_%i_spectrum_left_%i.npy' % (n_pts_fft,
                                                                               n_overlap,
                                                                               decimation_rate,
                                                                               spectrum_bins_left)
                                                 )
                        )
        self.lofar_data = lofar_data
        return lofar_data

    def _from_raw_data(self, raw_data, fs, verbose=0):
        decimation_rate = self.decimation_rate
        n_pts_fft = self.stftargs['n_pts_fft']
        n_overlap = self.stftargs['n_overlap']
        spectrum_bins_left = self.spectrum_bins_left

        lofar_data = list()
        lofar_trgt = list()
        for cls, runs in sorted(raw_data.items()):
            run_array = list()
            run_trgt = list()
            for run, run_data in sorted(runs.items()):
                power, freq, time = lofar(data=run_data,
                                         fs=fs[cls][run],
                                         n_pts_fft=n_pts_fft,
                                         n_overlap=n_overlap,
                                         decimation_rate=decimation_rate,
                                         spectrum_bins_left=spectrum_bins_left,
                                         **self.tpswargs)
                #lofar_data[cls] = dict()
                #lofar_data[cls][run] = power
                run_array.append(power)

                iclass= self.CLS_MAPPING[cls]
                run_trgt.append(np.repeat(iclass, power.shape[0]))
            lofar_data.append(np.concatenate(run_array, axis=0))
            lofar_trgt.append(np.concatenate(run_trgt))
        lofar_data = np.concatenate(lofar_data, axis=0)
        lofar_trgt = np.concatenate(lofar_trgt)
        return lofar_data, lofar_trgt, freq

    def from_chunk(self, data_chunk, fs, verbose=0):
        decimation_rate = self.decimation_rate
        n_pts_fft = self.stftargs['n_pts_fft']
        n_overlap = self.stftargs['n_overlap']
        spectrum_bins_left = self.spectrum_bins_left

        power, freq, time = lofar(data=data_chunk,
                                  fs=fs,
                                  n_pts_fft=n_pts_fft,
                                  n_overlap=n_overlap,
                                  decimation_rate=decimation_rate,
                                  spectrum_bins_left=spectrum_bins_left,
                                  **self.tpswargs)

        return power, freq, time


    def from_obj(self):
        pass