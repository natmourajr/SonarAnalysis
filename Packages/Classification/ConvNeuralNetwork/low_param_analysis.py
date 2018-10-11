import sys

import sklearn

from Functions.NpUtils.Scores import spIndex, recall_score, estimator_sp_index, estimator_recall_score, sp_index

sys.path.extend(['/home/pedrolisboa/Workspace/lps/LpsToolbox'])

from lps_toolbox.pipeline import ExtendedPipeline
from lps_toolbox.model_selection._search import PersGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Functions.ConvolutionalNeuralNetworks import MLPClassifier, ConvNetClassifier
from sklearn.model_selection import StratifiedKFold

from sklearn.datasets import load_iris
from keras.utils import to_categorical
from Functions.NpUtils.DataTransformation import Lofar2Image, SonarRunsInfo
from sklearn.externals import joblib
import os
# Database caracteristics
from Functions.SystemIO import load, save
import pandas as pd

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

all_data = dataset[0]
all_trgt = dataset[1]
class_labels = dataset[2]

# X,y = load_iris(True)
run_indices_info = SonarRunsInfo(audiodatapath + '/' + database)
cv = load('/home/pedrolisboa/Workspace/lps/SonarAnalysis/Results/10_folds_cv_runs_nested_4')


# mlp_cls = MLPClassifier(hidden_layer_sizes=(10, 3),
#                         activations=("tanh", "softmax"),
#                         input_shape=(4,))



scaler = StandardScaler()
lofar2image = Lofar2Image(all_data,all_trgt, 10, 10, run_indices_info, filepath=None, channel_axis='last')


cnn_clf = ConvNetClassifier(LofarObj=lofar2image,
                            n_filters=(6,),
                            conv_filter_sizes=((6,4),),
                            conv_strides=((3,2),),
                            conv_padding=("valid",),
                            conv_activations=("tanh",),
                            pool_filter_sizes=((2,4),),
                            pool_strides=((2,2),),
                            pool_padding=("valid",),
                            dense_layer_sizes=(10,4),
                            dense_activations=("tanh", "softmax"),
                            epochs=200)
# data, trgt, cv = lofar2image.gen_new_cv(all_data, all_trgt, cv)

# save(data, './data')
# save(trgt, './trgt')
# save(cv, './cv')
#
# raise NotImplementedError
pipe = ExtendedPipeline(steps=[('lofar2image', lofar2image),
                               # ('scaler', scaler),
                               ('clf', cnn_clf)],
                        memory=os.path.join(results_path, 'Low_Param_Analysis'))

def make_scorer(estimator, y, y_pred):
    new_y = estimator.transform()

gs = PersGridSearchCV(estimator=pipe,
                      param_grid={"lofar2image__window_size": [10, 15, 20, 25, 30]},
                      cv=cv,
                      verbose=1,
                      # scoring={'recall'},
                      # refit=False,
                      cachedir=os.path.join(results_path, 'Low_Param_Analysis'),
                      return_estimator=True)

# print list(cv.split(X,y))
# print list(cv.split(X,y))
# raise NotImplementedError
y = to_categorical(all_trgt, 4)
gs.fit(all_data, all_trgt, clf__validation_split=0.1, clf__class_weight=True, clf__n_inits=2, clf__verbose=1)
pd.DataFrame(gs.cv_results_).to_csv(os.path.join(results_path, 'Low_Param_Analysis/gs_results.csv'))
gs.best_estimator_._final_estimator.plotTraining(train_scores=['loss'],
                                                 val_scores=['val_loss'])

# import pandas as pd
# from pprint import pprint
# for column in pd.DataFrame(gs.cv_results_):
#     print pd.DataFrame(gs.cv_results_)[column]
