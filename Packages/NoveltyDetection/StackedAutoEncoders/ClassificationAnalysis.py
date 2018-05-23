import os
import sys
sys.path.insert(0,'..')

from noveltyDetectionConfig import CONFIG

import pickle
import numpy as np
import time
import argparse


# Argument Parser config

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("-l", "--layer", default=1, type=int, help="Select layer to be analyzed")
parser.add_argument("--type", type=str, default="representation",
                    help="Select type of analysis to be made <representation|classification>")
parser.add_argument("--analysis", type=str, default="mse/neurons_sweep",
                    help="Select analysis to be made. \n\
                    Representation:\n\
                     - mse/neurons_sweep \n\
                     - kl/neurons_sweep \n\
                    Classification: \n\
                     - sp \n\
                     - kl \n\
                     - sp/kl"
                     )

args = parser.parse_args()

analysis_type = ["representation",
                 "classification"]
analysis = {}
analysis["representation"] = ["mse/neurons_sweep",
                              "kl/neurons_sweep",
                             ]


analysis["classification"] = ["sp",
                              "sp/kl",
                             ]
try:
    analysis[args.type].index(args.analysis)
except:
    parser.print_help()
    exit()


from keras.utils import np_utils
from keras import backend as K

import sklearn.metrics
from sklearn.externals import joblib

from Functions import TrainParameters as trnparams
from Functions.StackedAutoEncoders import StackedAutoEncoders

import multiprocessing

init_time = time.time()

m_time = time.time()
print 'Time to import all libraries: '+str(m_time-init_time)+' seconds'

analysis_name = 'StackedAutoEncoder'

# Enviroment variables
data_path = CONFIG['OUTPUTDATAPATH']
results_path = CONFIG['PACKAGE_NAME']

# paths to export results
base_results_path = '%s/%s'%(results_path,analysis_name)
pict_results_path = '%s/pictures_files'%(base_results_path)
files_results_path = '%s/output_files'%(base_results_path)

# For multiprocessing purpose
num_processes = multiprocessing.cpu_count()

# Read data
m_time = time.time()

# Database caracteristics
database = '4classes'
n_pts_fft = 1024
decimation_rate = 3
spectrum_bins_left = 400
development_flag = False
development_events = 400

# Check if LofarData has created...
if not os.path.exists('%s/%s/lofar_data_file_fft_%i_decimation_%i_spectrum_left_%i.jbl'%
                      (data_path,database,n_pts_fft,decimation_rate,spectrum_bins_left)):
    print 'No Files in %s/%s\n'%(data_path,database)
else:
    #Read lofar data
    [data,trgt,class_labels] = joblib.load('%s/%s/lofar_data_file_fft_%i_decimation_%i_spectrum_left_%i.jbl'%
                                           (data_path,database,n_pts_fft,decimation_rate,spectrum_bins_left))


    m_time = time.time()-m_time
    print 'Time to read data file: '+str(m_time)+' seconds'

    # correct format
    all_data = data
    all_trgt = trgt

    # turn targets in sparse mode
    from keras.utils import np_utils
    trgt_sparse = np_utils.to_categorical(all_trgt.astype(int))

    # Process data
    # unbalanced data to balanced data with random data creation of small classes

    # Same number of events in each class
    qtd_events_biggest_class = 0
    biggest_class_label = ''

    for iclass, class_label in enumerate(class_labels):
        if sum(all_trgt==iclass) > qtd_events_biggest_class:
            qtd_events_biggest_class = sum(all_trgt==iclass)
            biggest_class_label = class_label
        print "Qtd event of %s is %i"%(class_label,sum(all_trgt==iclass))
    print "\nBiggest class is %s with %i events"%(biggest_class_label,qtd_events_biggest_class)

    balanced_data = {}
    balanced_trgt = {}

    from Functions import DataHandler as dh
    m_datahandler = dh.DataHandlerFunctions()

    for iclass, class_label in enumerate(class_labels):
        if development_flag:
            if len(balanced_data) == 0:
                class_events = all_data[all_trgt==iclass,:]
                balanced_data = class_events[0:development_events,:]
                balanced_trgt = (iclass)*np.ones(development_events)
            else:
                class_events = all_data[all_trgt==iclass,:]
                balanced_data = np.append(balanced_data,
                                          class_events[0:development_events,:],
                                          axis=0)
                balanced_trgt = np.append(balanced_trgt,(iclass)*np.ones(development_events))
        else:
            if len(balanced_data) == 0:
                class_events = all_data[all_trgt==iclass,:]
                balanced_data = m_datahandler.CreateEventsForClass(
                    class_events,qtd_events_biggest_class-(len(class_events)))
                balanced_trgt = (iclass)*np.ones(qtd_events_biggest_class)
            else:
                class_events = all_data[all_trgt==iclass,:]
                created_events = (m_datahandler.CreateEventsForClass(all_data[all_trgt==iclass,:],
                                                                     qtd_events_biggest_class-
                                                                     (len(class_events))))
                balanced_data = np.append(balanced_data,created_events,axis=0)
                balanced_trgt = np.append(balanced_trgt,
                                          (iclass)*np.ones(created_events.shape[0]),axis=0)

    all_data = balanced_data
    all_trgt = balanced_trgt

    # turn targets in sparse mode
    from keras.utils import np_utils
    trgt_sparse = np_utils.to_categorical(all_trgt.astype(int))

# Load train parameters
analysis_str = 'StackedAutoEncoder'
model_prefix_str = 'RawData'

trn_params_folder='%s/%s/%s_trnparams.jbl'%(results_path,analysis_str,analysis_name)
# if os.path.exists(trn_params_folder):
#     os.remove(trn_params_folder)
if not os.path.exists(trn_params_folder):
    trn_params = trnparams.NeuralClassificationTrnParams(n_inits=1,
                                                         hidden_activation='tanh', # others tanh, relu, sigmoid, linear
                                                         output_activation='linear',
                                                         n_epochs=500,  #500
                                                         patience=30,  #30
                                                         batch_size=128, #128
                                                         verbose=False,
                                                         optmizerAlgorithm='Adam',
                                                         metrics=['accuracy'],
                                                         loss='mean_squared_error')
    trn_params.save(trn_params_folder)
else:
    trn_params = trnparams.NeuralClassificationTrnParams()
    trn_params.load(trn_params_folder)

# Choose how many fold to be used in Cross Validation
n_folds = 10
CVO = trnparams.NoveltyDetectionFolds(folder=results_path, n_folds=n_folds, trgt=all_trgt, dev=development_flag, verbose=True)
print trn_params.get_params_str()

if development_flag:
    print '[+] Development mode'

# Train Process
if len(sys.argv) < 2:
    print '[-] Usage: %s <novelty class>'%sys.argv[0]
    exit()

inovelty = int(sys.argv[1])
print '\nNovelty class to train: %i'%inovelty

SAE = StackedAutoEncoders(params           = trn_params,
                          development_flag = development_flag,
                          n_folds          = n_folds,
                          save_path        = results_path,
                          CVO              = CVO,
                          noveltyDetection = True,
                          inovelty         = inovelty)


n_folds = len(CVO[inovelty])

hidden_neurons = range(400,0,-50) + [2]
print hidden_neurons

regularizer = "" #dropout / l1 / l2
regularizer_param = 0.5

trn_data = all_data[all_trgt!=inovelty]
trn_trgt = all_trgt[all_trgt!=inovelty]
trn_trgt[trn_trgt>inovelty] = trn_trgt[trn_trgt>inovelty]-1

# Choose layer to be trained
layer = 2

# Functions defined to be used by multiprocessing.Pool()

# SP Index

for ineuron in neurons_mat[:len(neurons_mat)-layer+2]:
        if ineuron == 0:
            ineuron = 1
        neurons_str = SAE[inovelty].getNeuronsString(all_data, hidden_neurons=hidden_neurons[:layer-1]+[ineuron])

        if verbose:
            print '[*] Layer: %i - Topology: %s'%(layer, neurons_str)

        def getSP(ifold):
            train_id, test_id = CVO[inovelty][ifold]

            # normalize known classes
            if trn_params.params['norm'] == 'mapstd':
                scaler = preprocessing.StandardScaler().fit(trn_data[inovelty][train_id,:])
            elif trn_params.params['norm'] == 'mapstd_rob':
                scaler = preprocessing.RobustScaler().fit(trn_data[inovelty][train_id,:])
            elif trn_params.params['norm'] == 'mapminmax':
                scaler = preprocessing.MinMaxScaler().fit(trn_data[inovelty][train_id,:])

            known_data = scaler.transform(trn_data[inovelty][test_id,:])

            classifier = SAE[inovelty].loadClassifier(data  = trn_data[inovelty],
                                                      trgt  = trn_trgt[inovelty],
                                                      hidden_neurons = hidden_neurons[:layer-1]+[ineuron],
                                                      layer = layer,
                                                      ifold = ifold)

            known_output = classifier.predict(known_data)

            num_known_classes = trn_trgt_sparse[inovelty].shape[1]

            efficiency = metrics.recall_score(trn_trgt_sparse[inovelty][test_id,:], np.round(known_output), average=None)
            sp_index = np.sum(efficiency)/num_known_classes * np.power(np.prod(efficiency), 1/num_known_classes)
            sp_index = np.sqrt(sp_index)

            return ifold, sp_index

start_time = time.time()

if K.backend() == 'theano':
    # Start Parallel processing
    p = multiprocessing.Pool(processes=num_processes)

    ####################### SAE LAYERS ############################
    # It is necessary to choose the layer to be trained

    # To train on multiple cores sweeping the number of folds
    # folds = range(len(CVO[inovelty]))
    # results = p.map(trainClassifierFold, folds)

    # To train multiple topologies sweeping the number of neurons
    # neurons_mat = range(0,400,50)
    # results = p.map(trainClassifierFold, neurons_mat)

    p.close()
    p.join()
else:
    neurons_mat = [10, 20] + range(50,450,50)
    # for ifold in range(len(CVO[inovelty])):
    #     result = trainFold(ifold)
    for ineuron in neurons_mat[:len(neurons_mat)-layer+2]:
        print '[*] Training Layer %i - %i Neurons'%(layer, ineuron)
        result = trainClassifierNeuron(ineuron)

end_time = time.time() - start_time
print "It took %.3f seconds to perform the training"%(end_time)
