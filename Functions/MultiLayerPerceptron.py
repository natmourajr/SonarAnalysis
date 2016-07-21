import os
import sys
import timeit

import numpy
import numpy as np

import random

import theano
import theano.tensor as T

from sklearn.cross_validation import ShuffleSplit
from sklearn import preprocessing


class InputLayer(object):
	""" Input Layer Class """
	
	def __init__(self, n_in, n_out, pre_function='mapstd', show=False):

		""" 
    	Params:
    	
    	n_in: integer
    	desc: dimensionality of input
    	
    	n_out: integer
    	desc: dimensionality of output
    	
    	pre_function: now NONE!!
    	desc: Pre-processing function
    	"""
    			
		if show:
			print ""
			print "Input Layer Constructor"
			print "n_in: ", n_in
			print "n_out: ", n_out
			print "pre_function: ", pre_function
			print ""

		self.n_in = n_in
		self.n_out = n_out
		self.pre_function = pre_function
		self.scaler = None
		
	def Show(self):
		print ""
		print "InputLayer Object"
		print "Input Dimension: ", self.n_in
		print "Output Dimension: ", self.n_out
		print "Pre-Processing Function: ", self.pre_function
		print ""
		
	def Train(self, input, show=False):
		if show: print "Input Layer Train - ", self.pre_function
		
		if self.pre_function == 'mapstd':
			self.scaler = preprocessing.StandardScaler().fit(input)
		if self.pre_function == 'mapminmax':
			self.scaler = preprocessing.MinMaxScaler().fit(input)
		if self.pre_function is None:
			self.scaler = None
	  
	def GetOutput(self,input, show=False):
		if show:
			print "InputLayer: GetOutput Function"
		
		# pre-process data
		#input = theano.shared(input,name="input")
		
		return (
		input if self.scaler is None
		else self.scaler.transform(input)
		)
		
					
class HiddenLayer(object):
	""" Hidden Layer Class """
	
	def __init__(self, rng, n_in, n_out, W=None, b=None, activation=T.tanh, show=False):
		
		""" 
    	Params:
    	
    	rng: numpy.random.RandomState
    	desc: a random number generator used to initialize weights
    	
    	n_in: integer
    	desc: dimensionality of input
    	
    	n_out: integer
    	desc: dimensionality of output
    	
    	activation: theano.Op or function
    	desc: Non linearity to be applied in this layer
    	"""
    	
		if show:
			print ""
			print "Hidden Layer Constructor"
			print "n_in: ", n_in
			print "n_out: ", n_out
			print "activation: ", activation
			print ""
		
		self.n_in = n_in
		self.n_out = n_out
		self.activation = activation
		
		
		# Initializing Weights
		
		# `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
		
		if W is None:
			W_values = numpy.asarray(rng.uniform(
			low=-numpy.sqrt(6. / (n_in + n_out)),
			high=numpy.sqrt(6. / (n_in + n_out)),
			size=(n_in, n_out)),
			dtype=theano.config.floatX)
			
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4
			
			# Note : optimal initialization of weights is dependent on the
        	#        activation function used (among other things).
        	#        For example, results presented in [Xavier10] suggest that you
        	#        should use 4 times larger initial weights for sigmoid
        	#        compared to tanh
        	#        We have no info for other function, so we use the same as
        	#        tanh.
			
			W = theano.shared(value=W_values, name='W', borrow=True)
		self.W = W
		self.W_init = W_values
		
		self.W_freeze = np.ones_like(W_values)
		
		
		# 'b' is initialized with 'b_values' which is a zero-vector
		
		if b is None:
			b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)
		self.b = b
		self.b_init = b_values
		
		self.b_freeze = np.ones_like(b_values)
		
		self.params = [self.W, self.b]
		self.freeze = [self.W_freeze, self.b_freeze]
		
	def Show(self):
		print ""
		print "HiddenLayer Object"
		print "Input Dimension: ", self.n_in
		print "Output Dimension: ", self.n_out
		print "Activation Function: ", self.activation
		print "Initialization Function: ", "Random"
		print ""

	def GetOutput(self, input, show=False):
		if show:
			print "Hidden Layer: GetOuput Function"
			
		lin_output = T.dot(input, self.W) + self.b
		return (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
    	
class OutputLayer(object):
	""" Output Layer Class """
	
	def __init__(self, rng, n_in, n_out, W=None, b=None, activation=T.tanh, show=False):
		
		""" 
    	Params:
    	
    	rng: numpy.random.RandomState
    	desc: a random number generator used to initialize weights
    	
    	n_in: integer
    	desc: dimensionality of input
    	
    	n_out: integer
    	desc: dimensionality of output
    	
    	activation: theano.Op or function
    	desc: Non linearity to be applied in this layer	
    	"""
    	
		if show:
			print ""
			print "Output Layer Constructor"
			print "n_in: ", n_in
			print "n_out: ", n_out
			print "activation: ", activation
			print ""

		self.n_in = n_in
		self.n_out = n_out
		self.W = W
		self.b = b
		self.activation = activation
		
		
		# Initializing Weights
		
		# `W` is initialized with `W_values` which is uniformely sampled
		# from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
		
		if W is None:
			W_values = numpy.asarray(rng.uniform(
			low=-numpy.sqrt(6. / (n_in + n_out)),
			high=numpy.sqrt(6. / (n_in + n_out)),
			size=(n_in, n_out)),
			dtype=theano.config.floatX)
			
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4
			
			# Note : optimal initialization of weights is dependent on the
        	#        activation function used (among other things).
        	#        For example, results presented in [Xavier10] suggest that you
        	#        should use 4 times larger initial weights for sigmoid
        	#        compared to tanh
        	#        We have no info for other function, so we use the same as
        	#        tanh.
			
			W = theano.shared(value=W_values, name='W', borrow=True)
		self.W = W
		self.W_init = W_values
		
		self.W_freeze = np.ones_like(W_values)
		
		
		# 'b' is initialized with 'b_values' which is a zero-vector
		
		if b is None:
			b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)
		self.b = b
		self.b_init = b_values
		
		self.b_freeze = np.ones_like(b_values)
		
		self.params = [self.W, self.b]
		self.freeze = [self.W_freeze, self.b_freeze]
		
	def Show(self):
		print ""
		print "OutputLayer Object"
		print "Input Dimension: ", self.n_in
		print "Output Dimension: ", self.n_out
		print "Activation Function: ", self.activation
		print "Initialization Function: ", "Random"
		print ""
		
	def GetOutput(self, input, show=False):
		if show:
			print "Output Layer: GetOuput Function"
		
		lin_output = T.dot(input, self.W) + self.b
		return (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
		
		
class TrainParameters(object):
	def __init__(self, perf_function='mse', l1reg=0.0, l2reg=0.0001, n_epochs=10, show_freq=1, show_precision=4, stop_criteria = 5, learning_rate=0.01, batch_size=10, itrn=None, itst=None, ival=None, perc_trn=0.5, perc_tst=0.25, show=False):
		""" Train Parameters Class 
			
			perf_function: Perfomance Function (MSE)
			
			l1reg: Size of L1 Regularization
			
			l2reg: Size of L2 Regularization
			
			n_epochs: Number of Epochs
			
			show_freq: Show Frequency in Training Process
			
			show_precision: Number of Decimals of Show Training
			
			stop_criteria: Number of Epoch of Stop Criteria
			
			learning_rate: Learning Rate
			
			batches_per_epochs = Number of batches in a Epoch 
			
			itrn: Train Indeces
			
			itst: Test Indeces
			
			ival: Validation Indeces
			
			perc_trn: Percentage of Events randomly choose for Train (used only if itrn is not defined)
			
			perc_tst: Percentage of Events randomly choose for Test (used only if itst is not defined)
			
		"""
		self.perf_function = perf_function
		self.l1reg = l1reg
		self.l2reg = l2reg
		self.n_epochs = n_epochs
		self.show_freq  = show_freq
		self.show_precision = show_precision
		self.stop_criteria = stop_criteria
		self.learning_rate = learning_rate
		self.batch_size = batch_size
			
		self.itrn = itrn
		self.itst = itst
		self.ival = ival
		
		if self.itrn is None:
			self.perc_trn = perc_trn
		else:
			self.perc_trn = len(self.itrn)/(len(self.itrn)+len(self.itst)+len(self.ival))
			
		if self.itst is None:
			self.perc_tst = perc_tst
		else:
			self.perc_tst = len(self.itst)/(len(self.itrn)+len(self.itst)+len(self.ival))
		
		if (self.perc_trn+self.perc_tst) > 1:
			print "\n\n***** Percentage Mistake!!! ***** \n\n"
		
		if (self.perc_trn+self.perc_tst) is 1:
			self.perc_val = self.perc_tst
		else:
			self.perc_val = 1-(self.perc_trn+self.perc_tst)
		
	def Show(self):
		print "Performance Function: ", self.perf_function
		print "L1 Regularization Value: ", self.l1reg
		print "L2 Regularization Value: ", self.l2reg
		print "Number of Epochs: ", self.n_epochs
		print "Frequency of Show: ",self.show_freq
		print "Show Precision: ",self.show_precision
		print "Stop Epochs: ",self.stop_criteria
		print "Learning Rate: ", self.learning_rate
		print "Batch Size: ", self.batch_size
		print "Train Indeces: ", self.itrn
		print "Test Indeces: ", self.itst
		print "Validation Indeces: ", self.ival

		print "Percentage of Train Events: ", (self.perc_trn if self.itrn is None
												else float(len(self.itrn))/
												(float(len(self.ival))+float(len(self.itst))
												+float(len(self.itrn)))
												)
		print "Percentage of Test Events: ", (self.perc_tst if self.itst is None
												else float(len(self.itst))/
												(float(len(self.ival))+float(len(self.itst))
												+float(len(self.itrn)))
												)
		print "Percentage of Validation Events: ", (self.perc_val if self.ival is None
													else float(len(self.ival))/
													(float(len(self.ival))+float(len(self.itst))
													+float(len(self.itrn)))
													)
		

		
class MLP(object):
    """ Multi-Layer Perceptron Class """
    
    def __init__(self, rng, n_in, n_hidden, n_out, hidden_act=T.tanh, output_act=T.tanh, show=False):
    	
    	if show:
    		print ""
    		print 'MLP Class Constructor'
    		print "n_in: ", n_in
    		print "n_hidden: ", n_hidden
    		print "n_out: ", n_out
    		print "hidden_act: ", hidden_act 
    		print "output_act: ", output_act 
    		print ""

    	""" 
    	Params:
    	
    	rng: numpy.random.RandomState
    	desc: a random number generator used to initialize weights
    	
    	n_in: integer
    	desc: dimensionality of input
    	
    	n_hidden: integer or array of integer
    	desc: dimensionality of hidden layer
    	
    	n_out: integer
    	desc: dimensionality of output
    	
    	hidden_act: theano.Op or function
    	desc: Non linearity to be applied in the hidden layer
    	
    	output_act: theano.Op or function
    	desc: Non linearity to be applied in the output layer
    	"""
    	
    	self.n_in = n_in
    	self.n_hidden = n_hidden
    	self.n_out = n_out
    	
    	self.InputLayer = InputLayer(n_in, n_in, pre_function='mapstd', show=show)
    	
    	self.HiddenLayer = {}
    	
    	for i in range(len(n_hidden)):
    		if i == 0:
    			self.HiddenLayer[i] = HiddenLayer(rng, n_in=n_in, n_out=n_hidden[i],show=show)
    		else:
    			self.HiddenLayer[i] = HiddenLayer(rng, n_in=n_hidden[i-1], n_out=n_hidden[i],show=show)
    			
    	self.OutputLayer = OutputLayer(rng,n_in=n_hidden[-1],n_out=n_out,show=show)
    	
    	params = []
    	for i in range(len(n_hidden)):
    		params = params + self.HiddenLayer[i].params
    	params = params + self.OutputLayer.params
    	
    	self.params = params

    	
    	freeze = []
    	for i in range(len(n_hidden)):
    		freeze = freeze + self.HiddenLayer[i].freeze
    	freeze = freeze + self.OutputLayer.freeze
    	
    	self.freeze = freeze

    	
    	
    	
    def Show(self):
    	""" This function show all relevant features of Neural Network Obj """
    
    	show_str = []
    	
    	# Title
    	print  "\nNeural Network Object "
    	self.InputLayer.Show()
    	for i in range(len(self.HiddenLayer)):
    		print "HiddenLayer["+str(i)+"]:"
    		self.HiddenLayer[i].Show()
    	self.OutputLayer.Show()
    	
    def GetOutput(self, input, show=False):
    	if show:
    		print "MLP: GetOutput Function"
    		
    	InputLayerOutput = self.InputLayer.GetOutput(input, show=show)
    	
    	HiddenLayerOutput = {}
    	
    	for i in range(len(self.n_hidden)):
    		if i == 0:
    			HiddenLayerOutput[i] = self.HiddenLayer[i].GetOutput(InputLayerOutput, show=show)
    		else:
    			HiddenLayerOutput[i] = self.HiddenLayer[i].GetOutput(HiddenLayerOutput[i-1], show=show)
    	
    	return self.OutputLayer.GetOutput(HiddenLayerOutput[len(self.n_hidden)-1],show=show)
    
    def GetL1Reg(self):
    	l1_value = 0
    	for i in range(len(self.n_hidden)):
    		l1_value = abs(self.HiddenLayer[i].W).sum()
    	return (l1_value + abs(self.OutputLayer.W).sum())
    	
    def GetL2Reg(self):
    	l2_value = 0
    	for i in range(len(self.n_hidden)):
    		l2_value = (self.HiddenLayer[i].W **2 ).sum()
    	return (l2_value + (self.OutputLayer.W **2 ).sum())

    
    def Performance(self, inputs, targets, perf_str, show=False):
    	if show:
    		print "Performance Function"
    	if perf_str == "mse":
    		return T.mean((targets-self.GetOutput(inputs)) ** 2)
    		
    	    	
    def Train(self, inputs, targets, trn_params=None):
    	print "Train Function"
    	
    	"""
    	Params: 
    	
    	input: vector of natural numbers
    	desc: Vector with inputs
    	
    	target: vector of natural numbers
    	desc: Vector with targets
    	
    	trn_params: Object of trn_params class
    	desc: Same as TrainParameters Class
    	"""
    	
    	self.InputLayer.Train(inputs)
    	inputs = self.InputLayer.GetOutput(inputs)
    	self.inputs = inputs
    	
    	
    	# check it (below)
    	self.targets = targets
    	
    	
    	# Processing data to be in Mini_batch Format
    	
    	if trn_params is None:
    		self.trn_params = TrainParameters()
    	else:
    		self.trn_params = trn_params
    		
    	
    	#print "Starting Train Process with the follow parameters:"
    	#self.trn_params.Show()
    	
    	# using Split from Sklearn
    	if self.trn_params.itrn == None:
    		ss = ShuffleSplit(len(targets), n_iter=1, test_size=trn_params.perc_tst)
    		for train, test in ss:
    			self.trn_params.itrn = train
    			self.trn_params.itst = test
    			self.trn_params.ival = test
    	
    	# Convert input variable to Theano mode
    	trn_inputs  = theano.shared(self.inputs[self.trn_params.itrn,:], borrow=True)
    	trn_targets = theano.shared(self.targets[self.trn_params.itrn,:], borrow=True)
    	
    	tst_inputs  = theano.shared(inputs[self.trn_params.itst,:], borrow=True)
    	tst_targets = theano.shared(targets[self.trn_params.itst,:], borrow=True)
    	
    	val_inputs  = theano.shared(inputs[self.trn_params.ival,:], borrow=True)
    	val_targets = theano.shared(targets[self.trn_params.ival,:], borrow=True)
    	
    	n_trn_batches = trn_inputs.get_value(borrow=True).shape[0]/self.trn_params.batch_size
    	n_tst_batches = tst_inputs.get_value(borrow=True).shape[0]/self.trn_params.batch_size
    	n_val_batches = val_inputs.get_value(borrow=True).shape[0]/self.trn_params.batch_size
    	
    	# allocate symbolic variables for the data
    	index = T.lscalar()  # index to a [mini]batch
    	sym_inputs  = T.matrix('sym_inputs')  # the data is presented as a symbolic variable
    	sym_targets = T.matrix('sym_targets') # the target is presented as a symbolic variable
    	
    	
    	cost = (
    	self.Performance(sym_inputs, sym_targets,trn_params.perf_function)
    	+ self.trn_params.l1reg*self.GetL1Reg() 
    	+ self.trn_params.l2reg*self.GetL2Reg()
    	)
    	
    	# compute the gradient of cost with respect to theta (sotred in params)
    	# the resulting gradients will be stored in a list gparams
    	gparams = [T.grad(cost, param) for param in self.params]
    	# specify how to update the parameters of the model as a list of
    	# (variable, update expression) pairs
    	
    	# given two lists of the same length, A = [a1, a2, a3, a4] and
    	# B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    	# element is a pair formed from the two lists :
    	#    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    	
    	# W' = W - learning_rate * gradient_cost
    	# b' = b - learning_rate * gradient_cost
    	
    	updates = [
    	(param, param - self.trn_params.learning_rate * gparam)
    	for param, gparam in zip(self.params, gparams)
    	]
    	
    	# Creating Train Model, Test Model and Validation Model
	# train model
    	train_model = theano.function(
        	inputs=[index],
        	outputs=[cost],
        	updates=updates,
        	givens={
            	sym_inputs: trn_inputs[index * self.trn_params.batch_size: (index + 1) * self.trn_params.batch_size,:],
            	sym_targets: trn_targets[index * self.trn_params.batch_size: (index + 1) * self.trn_params.batch_size,:]
        	}
    	)

    	# test model
    	test_model = theano.function(
        	inputs=[index],
        	outputs=cost,
        	givens={
            	sym_inputs: tst_inputs[index * self.trn_params.batch_size: (index + 1) * self.trn_params.batch_size,:],
            	sym_targets: tst_targets[index * self.trn_params.batch_size: (index + 1) * self.trn_params.batch_size,:]
        	}
    	)
    	# test model
    	val_model = theano.function(
        	inputs=[index],
        	outputs=cost,
        	givens={
            	sym_inputs: val_inputs[index * self.trn_params.batch_size: (index + 1) * self.trn_params.batch_size,:],
            	sym_targets: val_targets[index * self.trn_params.batch_size: (index + 1) * self.trn_params.batch_size,:]
        	}
    	)


    	
    	""" Train Process """
    	print "\n\nTraining..."
    	
    	epoch = 0
    	done_looping = False
    	
    	#set precision
    	numpy_opt = numpy.get_printoptions()
    	numpy.set_printoptions(precision=trn_params.show_precision)
    	
    	# Training Descriptor
    	trn_desc = {}
    	#trn_desc['perf'] = []
    	trn_desc['perf'] = numpy.zeros([trn_params.n_epochs,1])
    	trn_desc['tperf'] = numpy.zeros([trn_params.n_epochs,1])
    	trn_desc['vperf'] = numpy.zeros([trn_params.n_epochs,1])
    	
    	# stop criteria
    	best_val_cost = 9999
    	best_epoch = -1
    	best_mlp = []
    	
    	while (epoch < trn_params.n_epochs) and (not done_looping):
	  trn_cost = numpy.zeros([n_trn_batches,1])
	  grad_upd = numpy.zeros([n_trn_batches,1])
	  tst_cost = numpy.zeros([n_tst_batches,1])
	  val_cost = numpy.zeros([n_val_batches,1])
	  
	  for minibatch_index in xrange(n_trn_batches):
	    [trn_cost[minibatch_index]] = train_model(minibatch_index)
	    
	  for minibatch_index in xrange(n_tst_batches):
	    tst_cost[minibatch_index] = test_model(minibatch_index)

	  for minibatch_index in xrange(n_val_batches):
	    val_cost[minibatch_index] = val_model(minibatch_index)
	    
	  trn_desc['perf'][epoch] = numpy.mean(trn_cost)
	  trn_desc['tperf'][epoch] = numpy.mean(tst_cost)
	  trn_desc['vperf'][epoch] = numpy.mean(val_cost)
	  
	  epoch = epoch + 1
	  
	  if numpy.mean(val_cost) < best_val_cost:
	     best_epoch = epoch
	     best_mlp = self
	     best_val_cost = numpy.mean(val_cost)
	     
	  if (epoch - best_epoch) > trn_params.stop_criteria:
	      done_looping = True
	  
	  #print "Cost: ", minibatch_avg_cost
	  if float(epoch)%float(trn_params.show_freq) == 0:
	    print ("Epoch %04i - trn Cost: %02.3f - tst Cost: %02.3f - val Cost: %02.3f - Best Epoch: %04i - val Cost: %02.3f" % 
	    (epoch, numpy.mean(trn_cost), numpy.mean(tst_cost), numpy.mean(val_cost), best_epoch, best_val_cost))
	    
	if not(float(epoch)%float(trn_params.show_freq) == 0):
	  print "Final Epoch ", epoch," - trn Cost: ",numpy.mean(trn_cost) 
	
	# return to point before train
	numpy.set_printoptions(precision=numpy_opt['precision'])
	
	trn_desc['best_epoch'] = best_epoch
	trn_desc['best_val_cost'] = best_val_cost
	
	return trn_desc
	  
	