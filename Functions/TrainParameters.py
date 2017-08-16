"""
    This file contents some classification analysis functions
"""

import os
from sklearn.externals import joblib
from sklearn import cross_validation

class TrnParams(object):
    """
        Basic class
    """
    def __init__(self, analysis="None"):
        self.analysis = analysis
        self.params = None
        
    def save(self, name="None"):
        joblib.dump([self.params],name,compress=9)
        
    def load(self, name="None"):
        [self.params] = joblib.load(name)

        
# classification

def ClassificationFolds(folder,n_folds=2,trgt=None,dev=False, verbose=False):
    if n_folds < 2:
        print 'Invalid number of folds'
        return -1
    
    if not dev:
        file_name = '%s/%i_folds_cross_validation.jbl'%(folder,n_folds)
    else: 
        file_name = '%s/%i_folds_cross_validation_dev.jbl'%(folder,n_folds)
        
    if not os.path.exists(file_name):
        if verbose:
            print "Creating %s"%(file_name)
        
        if trgt is None:
            print 'Invalid trgt'
            return -1
        
        CVO = cross_validation.StratifiedKFold(trgt, n_folds)
        CVO = list(CVO)
        joblib.dump([CVO],file_name,compress=9)
    else:
        if verbose:
            print "File %s exists"%(file_name)
        [CVO] = joblib.load(file_name)
    
    return CVO

class NeuralClassificationTrnParams(TrnParams):
    """
        Neural Classification TrnParams
    """
    
    def __init__(self, n_inits=2,
                 norm='mapstd',
                 verbose=False,
                 train_verbose=False,
                 n_epochs=10,
                 learning_rate=0.01,
                 learning_decay=1e-6,
                 momentum=0.3,
                 nesterov=True,
                 patience=5,
                 batch_size=4,
                 hidden_activation='tanh',
                 output_activation='tanh'
                ):
        self.params = {}
        
        self.params['n_inits'] = n_inits
        self.params['norm'] = norm
        self.params['verbose'] = verbose
        self.params['train_verbose'] = train_verbose
        
        # train params
        self.params['n_epochs'] = n_epochs
        self.params['learning_rate'] = learning_rate
        self.params['learning_decay'] = learning_decay
        self.params['momentum'] = momentum
        self.params['nesterov'] = nesterov
        self.params['patience'] = patience
        self.params['batch_size'] = batch_size
        self.params['hidden_activation'] = hidden_activation
        self.params['output_activation'] = output_activation
        
    def get_params_str(self):
        param_str = ('%i_inits_%s_norm_%i_epochs_%i_batch_size_%s_hidden_activation_%s_output_activation'%
                     (self.params['n_inits'],self.params['norm'],self.params['n_epochs'],self.params['batch_size'],
                      self.params['hidden_activation'],self.params['output_activation']))
        return param_str
        