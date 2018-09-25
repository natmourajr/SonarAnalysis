import os
import sys
import time
from datetime import datetime, timedelta
sys.path.insert(0,'..')

from noveltyDetectionConfig import CONFIG
import numpy as np
import itertools
import multiprocessing
import pprint 

from SAENoveltyDetectionAnalysis import SAENoveltyDetectionAnalysis

num_processes = multiprocessing.cpu_count()

from Functions.telegrambot import Bot

my_bot = Bot("lisa_thebot")

# Enviroment variables
data_path = CONFIG['OUTPUTDATAPATH']
results_path = CONFIG['PACKAGE_NAME']

training_params = {
    "Technique": "StackedAutoEncoder",
    "TechniqueParameters": {
        "allow_change_weights": False #False
    },
    "DevelopmentMode": False,
    "DevelopmentEvents": 400,
    "NoveltyDetection": True,
    "InputDataConfig": {
        "database": "4classes",
        "n_pts_fft": 1024,
        "decimation_rate": 3,
        "spectrum_bins_left": 400,
        "n_windows": 1,
        "balance_data": True
    },
    "OptmizerAlgorithm": {
        "name": "Adam",
        "parameters": {
            "learning_rate": 0.001,
            "beta_1": 0.90,
            "beta_2": 0.999,
            "epsilon": 1e-08,
            "learning_decay": 1e-6,
            "momentum": 0.3,
            "nesterov": True
        }
    },
    "HyperParameters": {
        "n_folds": 4,#10, #4,
        "n_epochs": 500,#500,
        "n_inits": 1,#1, #2,
        "batch_size": 128,#128, #256,
        "kernel_initializer": "uniform",
        "bias_initializer": "ones", #zeros
        "encoder_activation_function": "tanh",
        "decoder_activation_function": "linear",
        "classifier_output_activation_function": "sigmoid",
        "norm": "mapstd",
        "metrics": ["accuracy"],
        "loss": "mean_squared_error",
        "dropout": False,
        "dropout_parameter": 0.00,
        "regularization": None, #None
        "regularization_parameter": 0.00 #0.00
    },
    "callbacks": {
        "EarlyStopping": {
            "patience": 30,
            "monitor": "val_loss"
        }
    }
}
analysis = SAENoveltyDetectionAnalysis(parameters=training_params,
                                       model_hash="",
                                       load_hash=False, load_data=True, verbose=True)
all_data, all_trgt, trgt_sparse = analysis.getData()

SAE = analysis.createSAEModels()

trn_data = analysis.trn_data
trn_trgt = analysis.trn_trgt
trn_trgt_sparse = analysis.trn_trgt_sparse

for inovelty in range(len(analysis.class_labels)):
    startTime = time.time()
    
    data=trn_data[inovelty]
    trgt=trn_trgt[inovelty]
    trainingType = "neuronSweep"
    analysis.train(layer=1,
                   inovelty=inovelty,
                   fineTuning=True,
                   trainingType=trainingType, #foldSweep, neuronSweep, normal, layerSweep
                   hidden_neurons=[400],
                   neurons_variation_step=100,
                   numThreads=5,
                   model_hash=analysis.model_hash)
    
    duration = str(timedelta(seconds=float(time.time() - startTime)))
    message = "Technique: Stacked Autoencoder\n"
    message = message + "Training Type: {}\n".format(trainingType)
    message = message + "Novelty Class: {}\n".format(analysis.class_labels[inovelty])
    message = message + "Hash: {}\n".format(analysis.model_hash)
    message = message + "Duration: {}\n".format(duration)
    try:
        my_bot.sendMessage(message)
    except Exception as e:
        print("Erro ao enviar mensagem. Erro: " + str(e))
    print "The training of the model for novelty class {0} took {1} to be performed\n".format(analysis.class_labels[inovelty], duration)