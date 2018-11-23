# coding=utf-8

import os
from itertools import product

from Functions.NpUtils.DataTransformation import SonarRunsInfo
from Functions.FunctionsDataVisualization import plotSpectrogram
from Functions.DataHandler import LofarDataset

datapath = os.getenv('OUTPUTDATAPATH')
audiodatapath = os.getenv('INPUTDATAPATH')
database = '4classes'

window_list = [2048, 1024, 512, 256, 128]
overlap_list = [0]
decimation_rate = 3
spectrum_bins_left_list = [820,400, 205, 103, 52]
lofar = LofarDataset(datapath)

for (window, spectrum_bins_left), overlap in product(zip(window_list, spectrum_bins_left_list), overlap_list):
    print('Window: %i  Overlap: %i' % (window, overlap))
    X, y, class_labels = lofar.loadData(database, window, overlap, decimation_rate, spectrum_bins_left)

    info = SonarRunsInfo(inputdatapath=os.path.join(audiodatapath, database), window=window)

    image_folder_path = os.path.join(datapath, 'LofarWaterfallImages', '%i_%i' % (window, overlap))
    for cls, runs in info.runs_named.items():
        print(cls)
        image_cls_path = os.path.join(image_folder_path, cls)
        if not os.path.exists(image_cls_path):
            os.makedirs(image_cls_path)
        for run_name, run_indices in runs.items():
            image_file_path = os.path.join(image_cls_path, run_name)
            if os.path.exists(os.path.join(image_file_path)):
                plotSpectrogram(X[run_indices], filename=image_file_path, colorbar=False)

