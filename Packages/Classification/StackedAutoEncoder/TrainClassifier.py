import os
import sys
import pickle
import numpy as np
import time

from keras.utils import np_utils
from keras.models import load_model

import sklearn.metrics
from sklearn.externals import joblib

from Functions import TrainParameters as trnparams
from Functions import TrainFunctions
from Functions.StackedAutoEncoders import StackedAutoEncoders

import multiprocessing

init_time = time.time()

m_time = time.time()
print 'Time to import all libraries: '+str(m_time-init_time)+' seconds'

analysis_name = 'StackedAutoEncoder'

# Enviroment variables
data_path = os.getenv('OUTPUTDATAPATH')
results_path = os.getenv('PACKAGE_NAME')

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
development_flag = True
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
            class_events = all_data[all_trgt==iclass,:]
            if len(balanced_data) == 0:
                balanced_data = class_events[0:development_events,:]
                balanced_trgt = (iclass)*np.ones(development_events)
            else:
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
os.remove(trn_params_folder)
if not os.path.exists(trn_params_folder):
    trn_params = trnparams.NeuralClassificationTrnParams(n_inits=1,
                                                         hidden_activation='tanh', # others tanh, relu, sigmoid, linear
                                                         output_activation='linear',
                                                         n_epochs=50,  #500
                                                         patience=10,  #30
                                                         batch_size=4, #256
                                                         verbose=False)
    trn_params.save(trn_params_folder)
else:
    trn_params = trnparams.NeuralClassificationTrnParams()
    trn_params.load(trn_params_folder)

# Choose how many folds to be used in Cross Validation
n_folds = 2
CVO = trnparams.ClassificationFolds(folder=results_path, n_folds=n_folds, trgt=all_trgt, dev=development_flag, verbose=False)
#print trn_params.get_params_str()

# Train Process
SAE = StackedAutoEncoders(params = trn_params,
                          development_flag = development_flag,
                          n_folds = n_folds,
                          save_path = results_path,
                          CVO = CVO)

# Choose layer to be trained
layer = 9

hidden_neurons = range(400,0,-50) + [2]
print hidden_neurons
# Functions defined to be used by multiprocessing.Pool()
def trainNeuron(ineuron):
    for ifold in range(n_folds):
        SAE.trainLayer(data=all_data,
                       trgt=all_trgt,
                       ifold=ifold,
                       hidden_neurons=hidden_neurons + [ineuron],
                       layer = layer)

def trainFold(ifold):
    return SAE.trainLayer(data=all_data,
                          trgt=all_trgt,
                          ifold=ifold,
                          hidden_neurons=hidden_neurons,
                          layer = layer)

# Train classifiers to their corresponding folds
def trainClassifierFold(ifold):
    return SAE.trainClassifier(data=all_data,
                        trgt = all_trgt,
                        ifold = 0,
                        hidden_neurons=hidden_neurons,
                        layer = 3)

# Train classifier sweeping the number of layer
def trainClassifierLayer(ilayer):
    for ifold in range(len(CVO)):
        SAE.trainClassifier(data=all_data,
                            trgt = all_trgt,
                            ifold = ifold,
                            hidden_neurons=hidden_neurons,
                            layer = ilayer)

start_time = time.time()

# Start Parallel processing
p = multiprocessing.Pool(processes=num_processes)

####################### SAE LAYERS ############################
# It is necessary to choose the layer to be trained

# To train on multiple cores sweeping the number of folds
# folds = range(len(CVO))
# results = p.map(trainFold, folds)

# To train multiple topologies sweeping the number of neurons
# neurons_mat = range(0,400,50) (start,final,step)
# results = p.map(trainNeuron, neurons_mat)

####################### CLASSIFIERS ############################
# It is necessary to choose the layer to be trained

# Train classifiers to their corresponding folds
# folds = range(len(CVO))
# results = p.map(trainClassifierFold, folds)

# Train classifier sweeping the number of layer
layers = range(1,10)
results = p.map(trainClassifierLayer, layers)

p.close()
p.join()

end_time = time.time() - start_time
print "It took %.3f seconds to perform the training"%(end_time)
