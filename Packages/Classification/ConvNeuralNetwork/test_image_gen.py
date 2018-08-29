import os
import sys
import pickle
import numpy as np
import time

from keras.utils import np_utils
from keras.models import load_model

import sklearn.metrics
from sklearn.externals import joblib
from Functions.DataHandler import NestedCV, lofar2image, SonarRunsCV

init_time = time.time()

m_time = time.time()
print 'Time to import all libraries: '+str(m_time-init_time)+' seconds'

# Enviroment variables
audiodatapath = os.getenv('INPUTDATAPATH')
data_path = os.getenv('OUTPUTDATAPATH')
results_path = os.getenv('PACKAGE_NAME')

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


ncv = NestedCV(5,10, results_path, audiodatapath)
ncv.loadCVs()
window = 20
stride = 10
x = np.array([1,2,3,4,5])
y = np.array([1,2,3])
print np.isin(y,x)
for cv_name, cv in ncv.cv.items():
    cv_info = SonarRunsCV(10, audiodatapath + '/' + database)
    print cv_name
    for train,test in cv:
        #x_train, y_train = lofar2image(data, trgt, train, class_labels, window, stride,
        #                               run_split_info=cv_info)
        x_test, y_test = lofar2image(data, trgt, test, class_labels, window, window,
                                     run_split_info=cv_info)
        print '\tIndices Train: %s Test: %s' % (train.shape, test.shape)
        #print '\tTrain: %s  Test : %s' % (x_train.shape, y_train.shape)
        print '\tTest: %s  Test : %s' % (x_test.shape, y_test.shape)
        #print x_test




