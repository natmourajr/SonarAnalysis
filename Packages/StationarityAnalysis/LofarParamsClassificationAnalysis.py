
# coding: utf-8

# ## Projeto Sonar
# ### Análise de estacionaridade para diferentes janelamentos. 
# #### Dataset: 4classes
# #### Autor: Pedro Henrique Braga Lisboa (pedrohblisboa@gmail.com)
# #### Laboratorio de Processamento de Sinais - UFRJ

# In[1]:


import sys
import os
import joblib
import numpy as np
import pandas as pd

from Functions.CrossValidation import SonarRunsCV

sys.path.extend(['/home/pedrolisboa/Workspace/lps/LpsToolbox'])
from itertools import starmap, product
from Functions.DataHandler import LofarDataset
from Functions.NpUtils.DataTransformation import SonarRunsInfo, lofar2image
from lps_toolbox.neural_network.classifiers import ConvNetClassifier
from lps_toolbox.metrics.classification import sp_index
#from Functions.ConvolutionalNeuralNetworks import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from itertools import repeat
datapath = os.getenv('OUTPUTDATAPATH')
audiodatapath = os.getenv('INPUTDATAPATH')
results_path = os.getenv('PACKAGE_NAME')
database = '4classes'


# In[2]:


# Load LOFAR data
#dataobj = LofarDataset(data_path=datapath)
def factor2(num):
    if num % 2 != 0:
        return [num // 2]
    factors = [num//2]
    factors.extend(factor2(num//2))
    return factors

#window_list = map(lambda e: pow(2,e), range(7,13,1))
#overlap_list = map(factor2, window_list)
window_list = [1024, 512, 256, 128]
overlap_list = [0]
decimation_rate = 3
spectrum_bins_left_list = [400, 205, 103, 52]


# In[3]:


def lofar_iter(estimator, train_fun, verbose):
    results = {'window': [],
               'overlap': [],
               'fold': [],
               'scores': []}
    for (window, spectrum_bins_left), overlap in product(zip(window_list, spectrum_bins_left_list), overlap_list):
        print window
        skf = SonarRunsCV(10, os.path.join(audiodatapath, database), window)
        if verbose:
            print('Window: %i  Overlap: %i' % (window, overlap))
        X, y, class_labels = lofar.loadData(database, window, overlap, decimation_rate, spectrum_bins_left)
        cvo_file = os.path.join(results_path, 
                                'db_%s_window_%i_overlap_%i_dec_%i_bins_%i_skf.jbl' % (database, 
                                                                                     window, 
                                                                                     overlap, 
                                                                                     decimation_rate, 
                                                                                     spectrum_bins_left))
        if os.path.exists(cvo_file):
            if verbose:
                print('\tLoading cross validation configuration')
            cvo = joblib.load(cvo_file)
        else:
            if verbose:
                print('\tCreating cross validation configuration')
            cvo = list(skf.split(X, y))
            joblib.dump(cvo, cvo_file)
        cachedir = os.path.join('FullClassification', cvo_file[:-4])

        s_info = SonarRunsInfo(os.path.join(audiodatapath, database), window)
        partial_results = train_fun(X,y, cvo, estimator, verbose, s_info, cachedir)

        window_l = list(repeat(window, len(partial_results['scores'])))
        overlap_l = list(repeat(overlap, len(partial_results['scores'])))
        print window_l
        results['window'].extend(window_l)
        results['overlap'].extend(overlap_l)
        for key in partial_results:
            print key
            results[key].extend(partial_results[key])
            
    return results
        
        


# In[4]:


from Functions.NpUtils.Scores import spIndex, recall_score
from sklearn.metrics import make_scorer
scoring = {'sp': spIndex}
scaler = StandardScaler()
import ipyparallel as ipp
import dill
# c = ipp.Client()
# c[:].use_dill()
# dview = c[:]
def novelty_detectionCV(X, y, cvo, estimator, verbose, s_info, cachedir):
    scores = list()
    fold = list()
    def train_fold(data):
        i_fold, train, test = data
        window_qtd=20
        window_qtd_stride=5
        print np.unique(y)

        X_train, y_train = lofar2image(X, y, train, window_qtd, window_qtd_stride, s_info)
        X_test, y_test = lofar2image(X, y, test, window_qtd, window_qtd, s_info)
        print np.unique(y_train)
        print np.unique(y_test)
        if verbose:
            print('\t\t Fold %i' % i_fold)
        # X_train = X[train]
        # y_train = y[train]
        #
        # X_test = X[test]
        # y_test = y[test]
        # scaler.fit(X_train, y_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)
        novelty_cls = 3
        X_train = X_train[y_train != novelty_cls]
        X_test = X_test[y_test != novelty_cls]
        y_train = y_train[y_train != novelty_cls]
        y_test = y_test[y_test != novelty_cls]
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        mask = np.ones(4, dtype=bool)
        mask[novelty_cls]=False
        if y_train.shape[1] == 4:
            y_train = y_train[:, mask]
            y_test = y_test[:, mask]
        elif y_train.shape[1] != 3:
            raise NotImplementedError

        print y_test.shape
        inner_cachedir = os.path.join(cachedir, 'ClassD', '%i_fold' % i_fold)
        estimator.fit(X_train, y_train,
                      validation_split=0.1,
#                      validation_data=(X_test, y_test),
                      n_inits=1,
                      verbose=verbose,
                      cachedir=inner_cachedir)
        print y_test.shape
        score = estimator.score(X_test, y_test)
        return (i_fold, score)
#         scores.append(score)
#         fold.append(i_fold)
    results = map(train_fold, [(i_fold, train, test) for i_fold, (train, test) in enumerate(cvo)])
    #results = dview.map_sync(train_fold, [(i_fold, train, test) for i_fold, (train, test) in enumerate(cvo)])
    fold,scores = map(list,zip(*results))
    return {'fold': fold,
            'scores': scores}
            
    


# In[ ]:


import time
verbose = 1
lofar = LofarDataset(datapath)
#skf = StratifiedKFold(n_splits=10)
estimator = ConvNetClassifier(conv_filter_sizes=((2, 10),),
                              conv_strides=((2, 5),),
                              conv_activations=("tanh",),
                              pool_filter_sizes=((2, 5),),
                              dense_layer_sizes=(10, 3),
                              dense_activations=("tanh", "softmax"),
                              epochs=50,
                              metrics=['acc']#, sp_index]
                              )
# estimator = mlpclassifier(layer_sizes=(10,4),
#                           activations=('tanh', 'softmax'),
#                           input_shape=(400,),
#                           epochs=100)
start = time.time()
results = lofar_iter(estimator, novelty_detectionCV, verbose)
stop = time.time()

print stop - start

import pandas as pd
#for key, value in results.items():
#    print key
#:wq    print len(value)
pd.DataFrame(results).to_csv('./results.csv')

