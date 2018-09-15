import os
from sklearn.externals import joblib

from Functions.CrossValidation import NestedCV
from Functions.TrainParameters import TrnParamsConvolutional
from Functions.TrainFunctions import ConvolutionTrainFunction
from Functions.ClassificationAnalysis import ModelDataCollection, CnnClassificationAnalysis

# Database caracteristics
datapath = os.getenv('OUTPUTDATAPATH')
audiodatapath = os.getenv('INPUTDATAPATH')
package_name = os.getenv('PACKAGE_NAME')

database = '4classes'
n_pts_fft = 1024
decimation_rate = 3
spectrum_bins_left = 400

# Check if LofarData has created...
if not os.path.exists('%s/%s/lofar_data_file_fft_%i_decimation_%i_spectrum_left_%i.jbl'%
                      (datapath,database,n_pts_fft,decimation_rate,spectrum_bins_left)):
    print 'No Files in %s/%s\n'%(datapath,database)
else:
    #Read lofar data
    [data, trgt, class_labels] = joblib.load('%s/%s/lofar_data_file_fft_%i_decimation_%i_spectrum_left_%i.jbl'%
                                              (datapath,database,n_pts_fft,decimation_rate,spectrum_bins_left))

ncv = NestedCV(5, 10, datapath, audiodatapath)
if ncv.exists():
    ncv.loadCVs()
else:
    ncv.createCVs(data, trgt)

# if not exists(datapath + '/' + 'iter_2/window_run'):
#     mkdir(datapath + '/' 'iter_2/window_run')

image_window = 10

trnparams = TrnParamsConvolutional(prefix="convnet_run_slide",
                                   input_shape=(image_window, 400, 1), epochs=30,
                                   layers=[
                                       ["Conv2D", {"filters": 6,
                                                   "kernel_size": (9, 5),
                                                   "strides": 1,
                                                   "data_format": "channels_last",
                                                   "padding": "valid"
                                                   }
                                        ],
                                       ["Activation", {"activation": "tanh"}],
                                       ["MaxPooling2D", {"pool_size": (2, 4),
                                                         "padding": "valid",
                                                         "strides": 1}
                                        ],
                                       ["Flatten", {}],
                                       ["Dense", {"units": 80}],
                                       ["Activation", {"activation": "tanh"}],
                                       ["Dense", {"units": 4}],
                                       ["Activation", {"activation": "softmax"}]
                                   ],
                                   loss="categorical_crossentropy")

#cnn_an = ModelDataCollection(ncv, trnparams, package_name, 'test_an', class_labels)
# cnn_an.fetchPredictions()
# cnn_an.fecthHistory()
# #cnn_an._reconstructPredictions(data, trgt, class_labels, image_window)
# cnn_an.plotConfusionMatrices()
# cnn_an.plotTraining()
# cnn_an.getScores()
# cnn_an.plotDensities()
# cnn_an.plotRuns(data, trgt, ["Conv2D"], overwrite=False)

#window_grid = [10, 20, 30, 40, 50]
window_grid = [40]
stride_grid = [10]
from itertools import product
param_mapping = dict()
window_grid = list(product(window_grid, stride_grid))
for (image_window, im_stride) in window_grid:
    trnparams = TrnParamsConvolutional(prefix="convnet/stride_%s" % im_stride,
                                       input_shape=(image_window, 400, 1), epochs=30,
                                       layers=[
                                           ["AveragePooling2D", {"strides": (4, 1),
                                                                 "pool_size": (4, 1),
                                                                 "padding": "valid"}],
                                           ["Conv2D", {"filters": 6,
                                                       "kernel_size": (9, 5),
                                                       "strides": 1,
                                                       "data_format": "channels_last",
                                                       "padding": "valid"
                                                       }
                                            ],
                                           ["Activation", {"activation": "tanh"}],
                                           ["MaxPooling2D", {"pool_size": (2, 4),
                                                             "padding": "valid",
                                                             "strides": 1}
                                            ],
                                           ["Flatten", {}],
                                           ["Dense", {"units": 80}],
                                           ["Activation", {"activation": "tanh"}],
                                           ["Dense", {"units": 4}],
                                           ["Activation", {"activation": "softmax"}]
                                       ],
                                       loss="categorical_crossentropy")

    param_mapping[str(image_window)] = trnparams

    model = ModelDataCollection(ncv, trnparams, package_name, '','iter_2/window_analysis_averaged/%i' % image_window, class_labels)
    #model._reconstructPredictions(data, trgt, image_window)
    #model.plotLayerOutputs(data,trgt,'MaxPooling2D')
    #model.plotRunsPredictions(data,trgt)

all_an = CnnClassificationAnalysis(ncv, param_mapping, package_name, '', 'iter_2/window_analysis_averaged', class_labels)
all_an.plotScores()
