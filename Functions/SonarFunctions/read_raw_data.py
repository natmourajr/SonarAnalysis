"""
    Projeto Marinha do Brasil

    Autor: Pedro Henrique Braga Lisboa (pedrohblisboa@lps.ufrj.br)
    Laboratorio de Processamento de Sinais - UFRJ
    Laboratorio de de Tecnologia Sonar - UFRJ/Marinha do Brasil
"""
from __future__ import print_function, division

import os
import getpass
import shutil
import socket
import warnings
from collections import OrderedDict
import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
# from librosa.core import load
import pyaudio
import queue
from threading import Thread

class AudioData:
    def __init__(self, inputpath, database):
        self.inputpath = inputpath
        self.database = database
        self.input_db_path = os.path.join(inputpath, database)

        self.raw_data = None
        self.data_fs = None
        self.datainfo = None

    def read_raw_data(self, verbose):
        if verbose:
            print('Reading Raw data in %s database\n' % self.database)

        datainfo = dict()
        datainfo['username'] = getpass.getuser()
        datainfo['computername'] = socket.gethostname()

        input_db_path = self.input_db_path
        class_folders = [folder for folder in os.listdir(input_db_path)
                         if not folder.startswith('.')]


        raw_data = dict()
        data_fs = dict()
        for cls_folder in class_folders:
            runfiles = os.listdir(os.path.join(input_db_path, cls_folder))
            if not runfiles:  # No files found inside the class folder
                if verbose:
                    print('Empty directory %s' % cls_folder)
                continue
            if verbose:
                print('Reading %s' % cls_folder)

            runfiles = os.listdir(os.path.join(input_db_path, cls_folder))
            audio_data = [(read_audio_file(os.path.join(input_db_path, cls_folder, runfile))) # read_audio returns a tuple
                           for runfile in runfiles]                # containing the signal and its sample rate

            cls_raw_data, cls_data_fs = zip(*audio_data)  # split audio data (raw_data) and sample rate into two arrays

            raw_data[cls_folder] = {runfile:rundata
                                    for runfile,rundata in zip(runfiles,cls_raw_data)}
            data_fs[cls_folder] = {runfile:fs
                                   for runfile, fs in zip(runfiles, cls_data_fs)}
        self.raw_data = raw_data
        self.data_fs = data_fs
        self.datainfo = datainfo
        return raw_data, data_fs

    def save_raw_data(self, outputpath, overwrite, savefmt='numpy', verbose=1):
        database = self.database
        raw_data, data_fs = self.raw_data, self.data_fs

        if raw_data is None and data_fs is None:
            if verbose:
                print('Raw data not found, performing read_raw_data...')
            raw_data, fs = self.read_raw_data(verbose)

        if savefmt.lower() not in ['numpy', 'txt']:
            raise warnings.warn('Invalid savefile format %s'
                                'Falling back to numpy .npy' % format)
        savefmt = savefmt.lower()

        out_db_path = os.path.exists(os.path.join(outputpath, database))

        if os.path.exists(out_db_path) and not overwrite:
            if not overwrite:
                if verbose:
                    print('Folder %s exits. To erase it, set overwrite to true')
                return
            else:
                shutil.rmtree(out_db_path)
                os.mkdir(out_db_path)

        for cls_folder in raw_data:
            out_cls_folder = os.path.join(out_db_path, cls_folder)
            os.mkdir(out_cls_folder)

            for runfile in raw_data[cls_folder]:
                run_data = raw_data[cls_folder][runfile]

                if savefmt == 'numpy':
                    np.save(os.path.join(out_cls_folder, 'raw_data_file.npy'),
                            run_data,
                            allow_pickle=False)  # Disabling pickle for portability purposes
                elif savefmt == 'txt':
                    np.savetxt(os.path.join(out_cls_folder, 'raw_data_file.txt'), delimiter=',', fmt='%.6f')


def read_audio_file(filepath):
    signal, fs = sf.read(filepath)
#     fs, signal= wav.read(filepath)
#     print(signal.max())
#     print(signal.min())
#     print(signal.dtype
#     signal, fs = load(filepath)
#     print(signal.max())
#     print(signal.min())
#     print(signal.dtype
#     signal = (np.array(signal, np.float) - 128)/128.0
#     signal = signal - signal.mean()
    return signal, fs

def threaded(fn):
    """To use as decorator to make a function call threaded.
    """
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


class AudioStream:
    def __init__(self, rate=22050,channels=1,width=1):
        self.buffer = queue.Queue()
        self.record_on = True
        self.stop_callback = False
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(width),
                        channels=channels,
                        rate=rate,
                        input=True,
                        stream_callback=self.callback)
        self.stream = stream
        self.start=True

    def callback(self, in_data, frame_count, time_info, status):
        if self.record_on:
            self.buffer.put(in_data)

        if self.stop_callback:
            callback_flag = pyaudio.paComplete
        else:
            callback_flag = pyaudio.paContinue

        return in_data, callback_flag

    @threaded
    def start(self):
        self.stream.start_stream()
        while self.start:
            continue
        self.start=True

    def stop(self):
        self.start=False
