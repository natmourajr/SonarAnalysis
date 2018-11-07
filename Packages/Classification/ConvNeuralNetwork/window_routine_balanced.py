import sys

from Functions.DataHandler import DataBalancer

sys.path.extend(['/home/pedrolisboa/Workspace/lps/SonarAnalysis/'])

from Functions.TrainFunctions import ConvolutionTrainFunction
from Functions.ConvolutionalNeuralNetworks import OldKerasModel
from Functions.TrainParameters import TrnParamsConvolutional
import os
from Functions.CrossValidation import NestedCV, SonarRunsCV
from Functions.NpUtils.DataTransformation import lofar2image, SonarRunsInfo
from Functions.SystemIO import exists, mkdir

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.externals import joblib

# Database caracteristics
datapath = os.getenv('OUTPUTDATAPATH')
audiodatapath = os.getenv('INPUTDATAPATH')
results_path = os.getenv('PACKAGE_NAME')

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

dataset = [data,trgt,class_labels]

ncv = NestedCV(5, 10, datapath, audiodatapath)
if ncv.exists():
    ncv.loadCVs()
else:
    ncv.createCVs(data, trgt)

if not exists(results_path + '/' + 'iter_2/window_run'):
    mkdir(results_path + '/' 'iter_2/window_run')

window_grid = [10, 20, 30, 40, 50]#, 60, 70, 80]
stride_grid = [10]
from itertools import product
window_grid = list(product(window_grid, stride_grid))
for cv_name, cv in ncv.cv.items():
    for (image_window, im_stride) in window_grid:
        trnparams = TrnParamsConvolutional(prefix="convnet_balanced/stride_%s" % im_stride,
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

        def transform_fn(all_data, all_trgt, index_info, info):
            run_info = SonarRunsInfo(audiodatapath + '/' + database)
            class_labels = {0:'ClassA',1:'ClassB',2:'ClassC',3:'ClassD'}
            if info == 'train':
                stride = im_stride
            else:
                stride = image_window
            x_test,y_test = lofar2image(all_data=all_data,
                                        all_trgt=all_trgt,
                                        index_info=index_info,
                                        window_size=image_window,
                                        stride=stride,
                                        run_indices_info=run_info)

            if info == 'train':
                db = DataBalancer()
                return db.oversample(x_test, y_test, class_labels,verbose=1)
            return x_test,y_test


        cvt = ConvolutionTrainFunction()
        cvt.loadModels([trnparams], OldKerasModel)
        cvt.loadData(dataset)
        cvt.loadFolds(cv)
        cvt.train(transform_fn=transform_fn, scaler=None,
                  fold_mode=cv_name, fold_balance='balanced',
                  verbose=(1, 1, 1))
