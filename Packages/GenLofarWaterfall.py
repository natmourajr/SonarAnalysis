# coding=utf-8

import os
import matplotlib
matplotlib.use('Agg')
from itertools import product

from Functions.NpUtils.DataTransformation import SonarRunsInfo
from Functions.FunctionsDataVisualization import plotSpectrogram
from Functions.DataHandler import LofarDataset

datapath = os.getenv('OUTPUTDATAPATH')
audiodatapath = os.getenv('INPUTDATAPATH')
database = '4classes'

window_list = [8192, 4096, 2048, 1024, 512, 256, 128]
overlap_list = window_list[1:]
overlap_list.append(64)
decimation_rate_list = [0, 3]
spectrum_bins_left_list = [3270, 1630, 820,400, 205, 103, 52]
lofar = LofarDataset(datapath)
param_list_w_overlap = list(product(zip(window_list, spectrum_bins_left_list, overlap_list), decimation_rate_list))
param_list_no_overlap = [((window, spectrum_bins_left, 0), decimation_rate)
                           for ((window, spectrum_bins_left, _), decimation_rate) in param_list_w_overlap]
param_list = param_list_no_overlap + param_list_w_overlap
print len(param_list)

for (window, spectrum_bins_left, overlap), decimation_rate in param_list:
    print('Window: %i  Overlap: %i  Decimation: %i' % (window, overlap, decimation_rate))
    X, y, class_labels = lofar.loadData(database, window, overlap, decimation_rate, spectrum_bins_left)

    info = SonarRunsInfo(inputdatapath=os.path.join(audiodatapath, database), window=window, overlap=overlap, decimation_rate=decimation_rate)

    image_folder_path = os.path.join(datapath, 'LofarWaterfallImages', 'window_%i_ovelap_%i_dec_rate_%i' % (window, overlap, decimation_rate))
    for cls, runs in info.runs_named.items():
        print(cls)
        image_cls_path = os.path.join(image_folder_path, cls)
        if not os.path.exists(image_cls_path):
            os.makedirs(image_cls_path)
        for run_name, run_indices in runs.items():
            image_file_path = os.path.join(image_cls_path, run_name)
            print 'Shape %s' % str(X.shape)
            if not os.path.exists(os.path.join(image_file_path)):
                print '%s %i' % (run_name, run_indices[-1])
                plotSpectrogram(X[run_indices], filename=image_file_path, colorbar=False)

