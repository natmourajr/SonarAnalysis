import os
import sys
import time
sys.path.insert(0,'..')

from noveltyDetectionConfig import CONFIG
import numpy as np

import multiprocessing

from SAENoveltyDetectionAnalysis import SAENoveltyDetectionAnalysis

num_processes = multiprocessing.cpu_count()

## Get the CLI arguments
####################################
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("-n", "--novelty", default=1, type=int, help="Select the novelty class")

args = parser.parse_args()

inovelty = args.novelty
#####################################
analysis_name = 'StackedAutoEncoder'

# Enviroment variables
data_path = CONFIG['OUTPUTDATAPATH']
results_path = CONFIG['PACKAGE_NAME']

analysis = SAENoveltyDetectionAnalysis(analysis_name="StackedAutoEncoder", development_flag = False, development_events=400)
all_data, all_trgt, trgt_sparse = analysis.getData()

analysis.setTrainParameters(n_inits=2,
                            hidden_activation='tanh',
                            output_activation='linear',
                            n_epochs=300,
                            n_folds=10,
                            patience=30,
                            batch_size=256,
                            verbose=False,
                            optmizerAlgorithm='Adam',
                            metrics=['accuracy'],
                            loss='mean_squared_error')

print ("\nResults path: " + analysis.getBaseResultsPath())

# trn_params = analysis.getTrainParameters()

analysis.createSAEModels()
SAE, trn_data, trn_trgt, trn_trgt_sparse = analysis.getSAEModels()

#for inovelty in range(len(analysis.class_labels)):

regularizer_values = [10**x for x in range(-1, 3)]

for regularizer in ['l1', 'l2']:
    for regularizer_param in regularizer_values:
        startTime = time.time()
        analysis.train(layer=1,
                       inovelty=inovelty,
                       ifold=10,
                       fineTuning=True,
                       trainingType="neuronSweep", #foldSweep, neuronSweep, normal
                       hidden_neurons=[400],
                       neurons_variation_step=25,
                       numThreads=8,
                       regularizer=regularizer,
                       regularizer_param=regularizer_param)
        print "The training of the model for novelty {0} took {1} seconds to be performed\n".format(analysis.class_labels[inovelty], time.time() - startTime)