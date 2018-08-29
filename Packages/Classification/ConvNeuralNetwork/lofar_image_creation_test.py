import sys
sys.path.extend(['/home/pedrolisboa/Workspace/lps/SonarAnalysis/'])

import os

from Functions.TrainParameters import TrnParamsConvolutional
from Functions.CrossValidation import NestedCV, SonarRunsCV
from Functions.NpUtils.DataTransformation import lofar2image, SonarRunsInfo
from Functions.SystemIO import exists, mkdir

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.externals import joblib

# Database caracteristics
datapath = os.getenv('OUTPUTDATAPATH')

audiodatapath = os.getenv('INPUTDATAPATH')
database = '4classes'
n_pts_fft = 1024
decimation_rate = 3
spectrum_bins_left = 400

# Check if LofarData has created...
if not os.path.exists('%s/%s/lofar_data_file_fft_%i_decimation_%i_spectrum_left_%i.jbl' %
                      (datapath, database, n_pts_fft, decimation_rate, spectrum_bins_left)):
    print 'No Files in %s/%s\n' % (datapath, database)
else:
    # Read lofar data
    dataset = joblib.load('%s/%s/lofar_data_file_fft_%i_decimation_%i_spectrum_left_%i.jbl' %
                                             (datapath, database, n_pts_fft, decimation_rate, spectrum_bins_left))

data = dataset[0]
trgt = dataset[1]
class_labels = dataset[2]

ncv = NestedCV(5, 10, datapath, audiodatapath)
if ncv.exists():
    ncv.loadCVs()
else:
    ncv.createCVs(data, trgt)

if not exists(datapath + '/' + 'iter_2/window_run'):
    mkdir(datapath + '/' 'iter_2/window_run')

window_grid = list(range(20,100,40))
image_stride = 10

for cv_name, cv in ncv.cv.items():
    for image_window in window_grid:
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
        run_info = SonarRunsInfo(audiodatapath + '/' + database)

        def transform_fn(all_data, all_trgt, index_info, info):
            if info == 'train':
                stride = image_stride
            else:
                stride = image_window
            return lofar2image(all_data=all_data,
                               all_trgt=all_trgt,
                               index_info=index_info,
                               window_size=image_window,
                               stride=stride,
                               run_indices_info=run_info)


        print 'Window size: %i' % image_window
        for fold_count, (train_index, test_index) in enumerate(cv):
            x_train, y_train = transform_fn(all_data=data, all_trgt=trgt,
                                            index_info=train_index, info='train')
            x_test, y_test = transform_fn(all_data=data, all_trgt=trgt,
                                          index_info=test_index, info='val')
            print 'Fold %i:' % fold_count
            print '\tX: %s \t Y: %s' % (x_train.shape, y_train.shape)
            print '\tX: %s \t Y: %s' % (x_test.shape, y_test.shape)

