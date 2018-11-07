""" Projeto Marinha do Brasil
    Autor: Pedro Henrique Braga Lisboa
    Laboratorio de Processamento de Sinais - UFRJ
    Laboratorio de Tecnologia Sonar - UFRJ/Marinha do Brasil
"""
import os
import joblib
from .lofar import lofar, tpsw

class LofarAnalysis:
    def __init__(self, decimation_rate, n_pts_fft, n_overlap, spectrum_bins_left='auto', **tpswargs):
        self.decimation_rate = decimation_rate
        self.stftargs = {'n_pts_fft': n_pts_fft,
                         'n_overlap': n_overlap}
        self.tpswargs = tpswargs
        print 'init'
        self.spectrum_bins_left = spectrum_bins_left

    def from_raw_data(self, raw_data, fs_data, database, outputpath=None, verbose=1):
        #input_db_path = os.path.join(inputpath, database)
        output_db_path = os.path.join(outputpath, database)

        decimation_rate = self.decimation_rate
        n_pts_fft = self.stftargs['n_pts_fft']
        n_overlap = self.stftargs['n_overlap']
        spectrum_bins_left = self.spectrum_bins_left
        print 'long before'
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
        self.lofar_data
        return lofar_data

    def _from_raw_data(self, raw_data, fs, verbose=0):
        decimation_rate = self.decimation_rate
        n_pts_fft = self.stftargs['n_pts_fft']
        n_overlap = self.stftargs['n_overlap']
        spectrum_bins_left = self.spectrum_bins_left

        lofar_data = dict()
        for cls, runs in raw_data.items():
            for run, run_data in runs.items():
                print 'before'
                power, _, _= lofar(data=run_data,
                                   fs=fs[cls][run],
                                   n_pts_fft=n_pts_fft,
                                   n_overlap=n_overlap,
                                   decimation_rate=decimation_rate,
                                   spectrum_bins_left=spectrum_bins_left,
                                   **self.tpswargs)

                lofar_data[cls][run] = power


    def from_stream(self, audio_stream, verbose=0):
        pass

    def from_obj(self):
        pass