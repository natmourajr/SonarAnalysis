import sys

sys.path.extend(['/home/pedrolisboa/Workspace/lps/LpsToolbox'])

from lps_toolbox.pipeline import ExtendedPipeline
from lps_toolbox.model_selection._search import PersGridSearchCV
from sklearn.preprocessing import StandardScaler

from Functions.ConvolutionalNeuralNetworks import ConvNetClassifier

from keras.utils import to_categorical
from Functions.NpUtils.DataTransformation import Lofar2Image, SonarRunsInfo
from sklearn.externals import joblib
import os
import keras.regularizers
# Database caracteristics
from Functions.SystemIO import load
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
lofar2image = Lofar2Image(all_data,all_trgt, 15, 10, run_indices_info, filepath=None, channel_axis='last',
                          memory=os.path.join(results_path, 'Low_Param_Analysis_Topology'))


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
                            epochs=100)
# data, trgt, cv = lofar2image.gen_new_cv(all_data, all_trgt, cv)

# save(data, './data')
# save(trgt, './trgt')
# save(cv, './cv')
#
# raise NotImplementedError
pipe = ExtendedPipeline(steps=[('lofar2image', lofar2image),
                               # ('scaler', scaler),
                               ('clf', cnn_clf)])#,
                        #memory=os.path.join(results_path, 'Low_Param_Analysis_Topology'))

import ipyparallel as ipp

c = ipp.Client(profile='ssh', sshserver='pedro.lisboa@ferney')

gs = PersGridSearchCV(estimator=pipe,
                      param_grid={"clf__kernel_regularizer": [None, keras.regularizers.l2()],
                                  "clf__dense_dropout": [None, (0.8,), (0.6,), (0.5,), (0.3,)],
                                  "clf__n_filters": [(6,), (9, )],
                                  "clf__conv_activations": [("tanh",)],
                                  "clf__dense_layer_sizes": [(10,4), (20, 4), (30, 4)]},
                      cv=cv,
                      verbose=1,
                      # scores={'spIndex': spIndex,
                      #         'eff_0': lambda x, y: recall_score(y, x)[0],
                      #         'eff_1': lambda x, y: recall_score(y, x)[1],
                      #         'eff_2': lambda x, y: recall_score(y, x)[2],
                      #         'eff_3': lambda x, y: recall_score(y, x)[3]},
                      cachedir=os.path.join(results_path, 'Low_Param_Analysis_Topology'),
                      client=c)

# print list(cv.split(X,y))
# print list(cv.split(X,y))
# raise NotImplementedError
y = to_categorical(all_trgt, 4)
gs.fit(all_data, all_trgt, clf__validation_split=0.1, clf__class_weight=True, clf__n_inits=2, clf__verbose=1)
pd.DataFrame(gs.cv_results_).to_csv(os.path.join(results_path, 'Low_Param_Analysis_Topology/gs_results.csv'))
gs.best_estimator_._final_estimator.plotTraining(train_scores=['loss'],
                                                 val_scores=['val_loss'])

# import pandas as pd
# from pprint import pprint
# for column in pd.DataFrame(gs.cv_results_):
#     print pd.DataFrame(gs.cv_results_)[column]
