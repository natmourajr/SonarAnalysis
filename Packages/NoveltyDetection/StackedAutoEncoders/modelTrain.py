import argparse

# Argument Parser config

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("-l", "--layer", default=1, type=int, help="Select layer to be trained")
parser.add_argument("-n", "--novelty", default=1, type=int, help="Select the novelty class")
parser.add_argument("-f", "--finetunning", default=0, type=int, help="Select if the training is to perform a fine tuning step")
parser.add_argument("-t", "--threads", default=8, type=int, help="Select the number of threads")
parser.add_argument("--regularizer", default="", type=str, help="Select the regularization")
parser.add_argument("--paramvalue", default=0.0, type=float, help="Select the regularization parameter value")
parser.add_argument("--type", type=str, default="representation",
                    help="Select type of training to be made <representation|finetuning>")

args = parser.parse_args()


from SAENoveltyDetectionAnalysis import SAENoveltyDetectionAnalysis
analysis = SAENoveltyDetectionAnalysis(analysis_name="StackedAutoEncoder", verbose=False)

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

analysis.createSAEModels()
SAE, trn_data, trn_trgt, trn_trgt_sparse = analysis.getSAEModels()

import multiprocessing
num_processes = args.threads

inovelty = args.novelty

fineTuning = args.finetunning == 1 if True else False

trainingType = args.type

data=trn_data[inovelty]
trgt=trn_trgt[inovelty]

hidden_neurons=range(400,0,-50)

neurons_mat = [10, 20] + range(50,450,50)

layer = args.layer

regularizer=args.regularizer
regularizer_param=args.paramvalue

if (trainingType == "normal"):
    if (fineTuning):
        for ifold in range (analysis.n_folds):
            # Do fine tuning training step for specified layer
            SAE[inovelty].trainClassifier(data=data, trgt=trgt, ifold=ifold, hidden_neurons=hidden_neurons, layer=layer, regularizer=regularizer,
                                          regularizer_param=regularizer_param)
    else:
        for ifold in range (analysis.n_folds):
            # Train autoencoder for specified layer
            SAE[inovelty].trainLayer(data=data, trgt=trgt, ifold=ifold, hidden_neurons=hidden_neurons, layer=layer,
                                     regularizer=regularizer,regularizer_param=regularizer_param)
elif (trainingType == "neuronSweep"):
    if (not neurons_mat):
        print "[-] Neurons array should not be empty for this type of training"
        exit()
    for ifold in range (analysis.n_folds):
        if (fineTuning):
            def train(ineuron):
                SAE[inovelty].trainClassifier(data  = data, trgt  = trgt, ifold = ifold, hidden_neurons = hidden_neurons[:layer-1] + [ineuron],
                                              layer = layer, regularizer=regularizer, regularizer_param=regularizer_param)

            p = multiprocessing.Pool(processes=num_processes)

            results = p.map(train, neurons_mat)

            p.close()
            p.join()
        else:
            def train(ineuron):
                SAE[inovelty].trainLayer(data  = data,
                                         trgt  = trgt,
                                         ifold = ifold,
                                         hidden_neurons = hidden_neurons[:layer-1] + [ineuron],
                                         layer = layer,
                                         regularizer=regularizer,
                                         regularizer_param=regularizer_param)
            p = multiprocessing.Pool(processes=num_processes)

            results = p.map(train, neurons_mat)

            p.close()
            p.join()
elif (trainingType == "layerSweep"):
    for ifold in range (analysis.n_folds):
        if (fineTuning):
            def train(ilayer):
                SAE[inovelty].trainClassifier(data  = data,
                                                   trgt  = trgt,
                                                   ifold = ifold,
                                                   hidden_neurons = hidden_neurons[:ilayer-1],
                                                   layer = ilayer,
                                                   regularizer=regularizer,
                                                   regularizer_param=regularizer_param)
            p = multiprocessing.Pool(processes=num_processes)

            results = p.map(train, range(1,layer+1))

            p.close()
            p.join()
        else:
            def train(ilayer):
                SAE[inovelty].trainLayer(data  = data,
                                              trgt  = trgt,
                                              ifold = ifold,
                                              hidden_neurons = hidden_neurons[:ilayer-1],
                                              layer = ilayer,
                                              regularizer=regularizer,
                                              regularizer_param=regularizer_param)
            p = multiprocessing.Pool(processes=num_processes)

            results = p.map(train, range(1,layer+1))

            p.close()
            p.join()
elif (trainingType == "foldSweep"):
    if (fineTuning):
        def train(fold):
            SAE[inovelty].trainClassifier(data  = data,
                                               trgt  = trgt,
                                               ifold = fold,
                                               hidden_neurons = hidden_neurons,
                                               layer = layer,
                                               regularizer=regularizer,
                                               regularizer_param=regularizer_param)
        p = multiprocessing.Pool(processes=num_processes)

        results = p.map(train, range(analysis.n_folds))

        p.close()
        p.join()
    else:
        def train(ilayer):
            SAE[inovelty].trainLayer(data  = data,
                                            trgt  = trgt,
                                            ifold = fold,
                                            hidden_neurons = hidden_neurons,
                                            layer = layer,
                                            regularizer=regularizer,
                                            regularizer_param=regularizer_param)
        p = multiprocessing.Pool(processes=num_processes)

        results = p.map(train, range(analysis.n_folds))

        p.close()
        p.join()
else:
    print "[-] %s is not set as a type of training"%trainingType
    exit()

print "[+] Training finished"