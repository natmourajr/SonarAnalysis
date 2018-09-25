import argparse
import multiprocessing

# Argument Parser config
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("-l", "--layer", default=1, type=int, help="Select layer to be trained")
parser.add_argument("-n", "--novelty", default=1, type=int, help="Select the novelty class")
parser.add_argument("-t", "--threads", default=8, type=int, help="Select the number of threads")
parser.add_argument("--hiddenNeurons", default="1", type=str, help="Select the hidden neurons configuration Ex.: (400x350x300)")
parser.add_argument("-k", "--modelhash", default="", type=str, help="Parameters Hash")
parser.add_argument("-s", "--neuronsVariationStep", default=50, type=int, help="Select the step to be used in neurons variation training")
parser.add_argument("-T", "--type", default="normal", type=str, help="Select the type of the training")
parser.add_argument("-v", "--verbose", default=False, type=bool, help="Verbose")

args = parser.parse_args()
num_processes = args.threads
inovelty = args.novelty
trainingType = args.type
step = args.neuronsVariationStep
layer = args.layer

from NNNoveltyDetectionAnalysis import NNNoveltyDetectionAnalysis

training_parameters = {"Technique": "NeuralNetwork"}

analysis = NNNoveltyDetectionAnalysis(
                                     parameters=training_parameters,
                                     model_hash=args.modelhash,
                                     load_hash=True,
                                     verbose=args.verbose
                                     )

trn_data = analysis.trn_data
trn_trgt = analysis.trn_trgt
trn_trgt_sparse = analysis.trn_trgt_sparse
models = analysis.models

hidden_neurons = [int(ineuron) for ineuron in args.hiddenNeurons.split('x')]
neurons_mat = [1] + range(step,hidden_neurons[layer-1]+step,step)

if (trainingType == "normal"):
    for ifold in range (analysis.n_folds):
        # Train autoencoder for specified layer
        models[inovelty].train(data=trn_data[inovelty],
                               trgt=trn_trgt[inovelty],
                               ifold=ifold,
                               hidden_neurons=hidden_neurons,
                               layer=layer
                               )
elif (trainingType == "neuronSweep"):
    if (not neurons_mat):
        print "[-] Neurons array should not be empty for this type of training"
        exit()
    for ineuron in neurons_mat:
        def train(ifold):
            models[inovelty].train(data  = trn_data[inovelty],
                                   trgt  = trn_trgt[inovelty],
                                   ifold = ifold,
                                   hidden_neurons = hidden_neurons[:layer-1] + [ineuron],
                                   layer = layer
                                   )
        p = multiprocessing.Pool(processes=num_processes)

        results = p.map(train, range(analysis.n_folds))

        p.close()
        p.join()
elif (trainingType == "layerSweep"):
    for ifold in range (analysis.n_folds):
        def train(ilayer):
            models[inovelty].train(data  = trn_data[inovelty],
                                   trgt  = trn_trgt[inovelty],
                                   ifold = ifold,
                                   hidden_neurons = hidden_neurons[:ilayer-1],
                                   layer = ilayer
                                   )
        p = multiprocessing.Pool(processes=num_processes)

        results = p.map(train, range(1,layer+1))

        p.close()
        p.join()
elif (trainingType == "foldSweep"):
    def train(fold):
        models[inovelty].train(data  = trn_data[inovelty],
                               trgt  = trn_trgt[inovelty],
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