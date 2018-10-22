
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
#sys.path.extend(['/home/pedrolisboa/Workspace/lps/LpsToolbox'])
from itertools import starmap
from Functions.DataHandler import LofarDataset
from Functions.ConvolutionalNeuralNetworks import MLPClassifier
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
window_list = [1024]
overlap_list = [0]
decimation_rate = 3
spectrum_bins_left = 400


# In[3]:


def lofar_iter(estimator, fold_fun, dataset_obj, train_fun, verbose):
    results = {'window': [],
               'overlap': [],
               'fold': [],
               'scores': []}
    for window, overlap in zip(window_list, overlap_list):
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
        cachedir = cvo_file[:-4]
        partial_results = train_fun(X,y, cvo, estimator, verbose, cachedir)
        
        results['window'] = list(repeat(window, len(partial_results['scores'])))
        results['overlap'] = list(repeat(overlap, len(partial_results['scores'])))
        for key in partial_results:
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
def novelty_detectionCV(X, y, cvo, estimator, verbose, cachedir):
    scores = list()
    fold = list()
    def train_fold(data):
        i_fold, train, test = data
        if verbose:
            print('\t\t Fold %i' % i_fold)
        X_train = X[train]
        y_train = y[train]

        X_test = X[test]
        y_test = y[test]
        scaler.fit(X_train, y_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        estimator.cachedir = os.path.join(cachedir, '%i_fold' % i_fold)
        estimator.fit(X_train, y_train,
                      validation_split=0.1,
                      n_inits=10,
                      verbose=verbose)
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
verbose = 3
lofar = LofarDataset(datapath)
skf = StratifiedKFold(n_splits=10)
estimator = MLPClassifier(layer_sizes=(10,4),
                          activations=('tanh', 'softmax'),
                          input_shape=(400,),
                          epochs=100)
start = time.time()
results = lofar_iter(estimator, skf, lofar, novelty_detectionCV, verbose)
stop = time.time()

print stop - start


# In[ ]:


import pandas as pd
pd.DataFrame(results)
# results['window'] = list(results['window'])
# results['overlap'] = list(results['overlap'])


# In[12]:


import ipyparallel as ipp
c = ipp.Client()


# In[13]:


c.ids


# In[ ]:


c[:].map_sync()

