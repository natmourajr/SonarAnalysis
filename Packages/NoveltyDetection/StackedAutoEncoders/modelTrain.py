import argparse

# Argument Parser config

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("-l", "--layer", default=1, type=int, help="Select layer to be trained")
parser.add_argument("-n", "--novelty", default=1, type=int, help="Select the novelty class")
parser.add_argument("-f", "--finetunning", default=0, type=int, help="Select if the training is to perform a fine tuning step")
parser.add_argument("-t", "--threads", default=8, type=int, help="Select the number of threads")
parser.add_argument("--hiddenNeurons", default="1", type=str, help="Select the hidden neurons configuration Ex.: (400x350x300)")
parser.add_argument("--developmentEvents", default=400, type=int, help="Select the number of events to development")
parser.add_argument("--developmentFlag", default=0, type=int, help="Turn the development mode on")
parser.add_argument("-h", "--hash", default="", type=str, help="Parameters Hash")
parser.add_argument("--type", type=str, default="representation",
                    help="Select type of training to be made <representation|finetuning>")
parser.add_argument("-s", "--neuronsVariationStep", default=50, type=int, help="Select the step to be used in neurons variation training")

args = parser.parse_args()

from SAENoveltyDetectionAnalysis import SAENoveltyDetectionAnalysis

analysis = SAENoveltyDetectionAnalysis(
                                        model_hash = args.hash,
                                        load_hash  = (hash != "")
                                       )

analysis.loadTrainParametersByHash(args.hash)
analysis.createSAEModels()

SAE = analysis.getSAEModels()
trn_data = analysis.trn_data
trn_trgt = analysis.trn_trgt
trn_trgt_sparse = analysis.trn_trgt_sparse

import multiprocessing
num_processes = args.threads

inovelty = args.novelty

fineTuning = args.finetunning

trainingType = args.type

step = args.neuronsVariationStep

data=trn_data[inovelty]
trgt=trn_trgt[inovelty]

layer = args.layer

hidden_neurons = [int(ineuron) for ineuron in args.hiddenNeurons.split('x')]
neurons_mat = [1] + range(step,hidden_neurons[layer-1]+step,step)

if (trainingType == "normal"):
    if (fineTuning):
        for ifold in range (analysis.n_folds):
            # Do fine tuning training step for specified layer
            SAE[inovelty].trainClassifier(data=data, trgt=trgt, ifold=ifold, hidden_neurons=hidden_neurons, layer=layer)
    else:
        for ifold in range (analysis.n_folds):
            # Train autoencoder for specified layer
            SAE[inovelty].trainLayer(data=data, trgt=trgt, ifold=ifold, hidden_neurons=hidden_neurons, layer=layer)
elif (trainingType == "neuronSweep"):
    if (not neurons_mat):
        print "[-] Neurons array should not be empty for this type of training"
        exit()
    for ineuron in neurons_mat:
        if (fineTuning):
            def train(ifold):
                SAE[inovelty].trainClassifier(data  = data,
                                              trgt  = trgt,
                                              ifold = ifold,
                                              hidden_neurons = hidden_neurons[:layer-1] + [ineuron],
                                              layer = layer
                                              )

            p = multiprocessing.Pool(processes=num_processes)

            results = p.map(train, range(analysis.n_folds))

            p.close()
            p.join()
        else:
            def train(ifold):
                SAE[inovelty].trainLayer(data  = data,
                                         trgt  = trgt,
                                         ifold = ifold,
                                         hidden_neurons = hidden_neurons[:layer-1] + [ineuron],
                                         layer = layer)
            p = multiprocessing.Pool(processes=num_processes)

            results = p.map(train, range(analysis.n_folds))

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
                                               layer = ilayer
                                              )
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
                                          layer = ilayer
                                         )
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
                                           layer = layer
                                          )
        p = multiprocessing.Pool(processes=num_processes)

        results = p.map(train, range(analysis.n_folds))

        p.close()
        p.join()
    else:
        def train(fold):
            SAE[inovelty].trainLayer(data  = data,
                                    trgt  = trgt,
                                    ifold = fold,
                                    hidden_neurons = hidden_neurons,
                                    layer = layer
                                    )
        p = multiprocessing.Pool(processes=num_processes)

        results = p.map(train, range(analysis.n_folds))

        p.close()
        p.join()
else:
    print "[-] %s is not set as a type of training"%trainingType
    exit()

print "[+] Training finished"