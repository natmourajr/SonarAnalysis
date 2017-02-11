""" 
  This file contents some processing functions
"""

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras.callbacks as callbacks
from keras.utils import np_utils
from keras.layers import Merge

from keras import backend as K

class TrnParams(object):
	def __init__(self, learning_rate=0.01, 
	learning_decay=1e-6, momentum=0.3, 
	nesterov=True, train_verbose=False, verbose= False, 
	n_epochs=500,batch_size=8):
		self.learning_rate = learning_rate
		self.learning_decay = learning_decay
		self.momentum = momentum
		self.nesterov = nesterov
		self.train_verbose = train_verbose
		self.verbose = verbose
		self.n_epochs = n_epochs
		self.batch_size = batch_size
	
	def Print(self):
		print 'Class TrnParams'
		print 'Learning Rate: %1.5f'%(self.learning_rate)
		print 'Learning Decay: %1.5f'%(self.learning_decay)
		print 'Momentum: %1.5f'%(self.momentum)
		if self.nesterov:
			print 'Nesterov: True'
		else:
			print 'Nesterov: False'
		if self.verbose:
			print 'Verbose: True'
		else:
			print 'Verbose: False'
		
		if self.train_verbose:
			print 'Train Verbose: True'
		else:
			print 'Train Verbose: False'
		print 'NEpochs: %i'%(self.n_epochs)
		print 'Batch Size: %i'%(self.batch_size)	
	

class PCDIndependent(object):

	"""
	PCD Independent class
		This class implement the Principal Component of Discrimination Analysis in Independent Approach
	"""		
	def __init__(self, n_components=2):
		"""
		PCD Independent constructor
			n_components: number of components to be extracted
		"""
		self.n_components = n_components
		self.models = {}
		self.trn_descs = {}
		self.pcds = {}
		
	def fit(self, data, target, train_ids, test_ids, trn_params=None):
		"""
		PCD Independent fit function
			data: data to be fitted (events x features)
			target: class labels - sparse targets (events x number of classes)

			train_ids:  train indexes - user generated
			test_ids: test indexes - user generated
			trn_params: class TrnParams (optional)
		"""		
		print 'PCD Independent fit function'

		if trn_params is None:
			trn_params = TrnParams()
		
		#print 'Train Parameters'	
		#trn_params.Print()
		
		if trn_params.verbose: 
			print 'PCD Independent Model Struct: %i - %i - %i'%(data.shape[1],1,target.shape[1])

		for ipcd in range(self.n_components):
			print 'Training %i PCD of %i PCDs'%(ipcd+1,self.n_components)
			
			if ipcd == 0:
				my_model = Sequential()
				
				# add a linear layer to isolate the input of NN model
				my_model.add(Dense(data.shape[1],input_dim=data.shape[1], init='identity',trainable=False))
				
				my_model.add(Activation('linear'))
				
				# add a non-linear single neuron layer to compress all information
				my_model.add(Dense(1, input_dim=data.shape[1], init='uniform'))
				my_model.add(Activation('tanh'))
				
				# add a non-linear output layer with max sparse target shape
				my_model.add(Dense(target.shape[1], init='uniform')) 
				my_model.add(Activation('tanh'))
				
				# creating a optimization function using steepest gradient
				sgd = SGD(lr=trn_params.learning_rate, decay=trn_params.learning_decay, 
                          		  momentum=trn_params.momentum, nesterov=trn_params.nesterov)
				
				# compile the model
				my_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy','mean_squared_error'])
				
				# early stopping to avoid overtraining
				earlyStopping = callbacks.EarlyStopping(monitor='val_loss', 
                                                        patience=25, verbose=0, 
                                                        mode='auto')
				
				# Train model
				init_trn_desc = my_model.fit(data[train_ids], target[train_ids],
                                          		  nb_epoch=trn_params.n_epochs, 
                                          		  batch_size=trn_params.batch_size,
                                          		  callbacks=[earlyStopping], 
                                          		  verbose=trn_params.train_verbose,
                                          		  validation_data=(data[test_ids],
                                                          		   target[test_ids]),
                                          		  shuffle=True)
				self.models[ipcd] = my_model
				self.trn_descs[ipcd] = init_trn_desc
                		self.pcds[ipcd] = my_model.layers[2].get_weights()[0]
				if trn_params.verbose : 
					print 'PCD %i - Train process is done, val_cost: %1.5f'%(ipcd+1,np.min(init_trn_desc.history['val_loss']))
			else:
				my_model = Sequential()
				
				# add a linear layer to isolate the input of NN model
				my_model.add(Dense(data.shape[1],input_dim=data.shape[1], init='identity',trainable=False))
				
				my_model.add(Activation('linear'))
				
				# add a non-linear single neuron layer to compress all information
				my_model.add(Dense(1, input_dim=data.shape[1], init='uniform'))
				my_model.add(Activation('tanh'))
				
				# add a non-linear output layer with max sparse target shape
				my_model.add(Dense(target.shape[1], init='uniform')) 
				my_model.add(Activation('tanh'))
			
				# creating a optimization function using steepest gradient
				sgd = SGD(lr=trn_params.learning_rate, decay=trn_params.learning_decay, 
                          		  momentum=trn_params.momentum, nesterov=trn_params.nesterov)
				
				# compile the model
				my_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy','mean_squared_error'])
				
				# early stopping to avoid overtraining
				earlyStopping = callbacks.EarlyStopping(monitor='val_loss', 
                                                        patience=25, verbose=0, 
                                                        mode='auto')

				# remove the projection of previous extracted pcds from random init weights
				w = my_model.layers[2].get_weights()[0]
				w_proj = np.zeros_like(w)
				
				# loop over previous pcds				
				for i_old_pcd in range(ipcd):
					w_proj = (w_proj + (np.inner(np.inner(self.pcds[i_old_pcd],w),self.pcds[i_old_pcd].T)/
							    np.inner(self.pcds[i_old_pcd].T,self.pcds[i_old_pcd].T)))
				w_remove_proj = w - w_proj
				weights = my_model.layers[2].get_weights()
				weights[0] = w_remove_proj
				my_model.layers[2].set_weights(weights)

				# remove the projection of previous extracted pcds from data
				data_proj = np.zeros_like(data)

				# loop over previous pcds
				for i_old_pcd in range(ipcd-1):
					data_proj = (data_proj + (np.inner(np.inner(self.pcds[i_old_pcd].T,data).T,self.pcds[i_old_pcd])/
								  np.inner(self.pcds[i_old_pcd].T,self.pcds[i_old_pcd].T)))
				
				data_with_proj = data - data_proj			         
				
				# Train model
				init_trn_desc = my_model.fit(data_with_proj[train_ids], target[train_ids],
                                          		  nb_epoch=trn_params.n_epochs, 
                                          		  batch_size=trn_params.batch_size,
                                          		  callbacks=[earlyStopping], 
                                          		  verbose=trn_params.train_verbose,
                                          		  validation_data=(data[test_ids],
                                                          		   target[test_ids]),
                                          		  shuffle=True)
				self.models[ipcd] = my_model
				self.trn_descs[ipcd] = init_trn_desc
                		self.pcds[ipcd] = my_model.layers[2].get_weights()[0]
				if trn_params.verbose:
					print 'PCD %i - Train process is done, val_cost: %1.5f'%(ipcd+1,np.min(init_trn_desc.history['val_loss']))
		return self

	def transform_with_activation_function(self, data):
		"""
			PCD Independent auxiliar transform function
				data: data to be transformed (events x features)
		"""
		output = []
		for ipcd in range(self.n_components):
			# get the output of an intermediate layer		
			# with a Sequential model
			get_layer_output = K.function([self.models[ipcd].layers[0].input],[self.models[ipcd].layers[3].output])
			pcd_output = get_layer_output([data])[0]
			if ipcd == 0:
				output = pcd_output
			else:
				output = np.append(output,pcd_output, axis=1)
		return output

	def transform_without_activation_function(self, data):
		"""
			PCD Independent auxiliar transform function
				data: data to be transformed (events x features)
		"""

		output = []
		for ipcd in range(self.n_components):
			# get the output of an intermediate layer		
			# with a Sequential model
			get_layer_output = K.function([self.models[ipcd].layers[0].input],[self.models[ipcd].layers[2].output])
			pcd_output = get_layer_output([data])[0]
			if ipcd == 0:
				output = pcd_output
			else:
				output = np.append(output,pcd_output, axis=1)
		return output

	def transform(self, data, use_activation=False):
		"""
			PCD Independent transform function
				data: data to be transformed (events x features)
				use_activation: boolean to use or not the activation function
		"""
		if use_activation:
			return self.transform_with_activation_function(data)
		else:
			return self.transform_without_activation_function(data)
	
	def get_degree_matrix(self):
		degree_matrix = np.zeros([self.n_components,self.n_components])

		if self.models == {}:
			return degree_matrix
		
		for ipcd in range(self.n_components):
			for jpcd in range(self.n_components):
				degree = (np.inner(self.pcds[ipcd].T,self.pcds[jpcd].T)/
					 (np.linalg.norm(self.pcds[ipcd])*np.linalg.norm(self.pcds[jpcd])))
				degree = round(degree.real,6)
				degree = np.arccos(degree)
				degree = np.degrees(degree)
				if degree > 90 and degree < 180:
					degree = degree - 180
				if degree > 180 and degree < 270:
					degree = degree - 180
				degree_matrix[ipcd,jpcd] = np.abs(degree)
		return degree_matrix

class PCDCooperative(object):

	"""
	PCD Cooperative class
		This class implement the Principal Component of Discrimination Analysis in Cooperative Approach
	"""		
	def __init__(self, n_components=2):
		"""
		PCD Cooperative constructor
			n_components: number of components to be extracted
		"""
		self.n_components = n_components
		self.models = {}
		self.trn_descs = {}
		self.pcds = {}
		
	def fit(self, data, target, train_ids, test_ids, trn_params=None):
		"""
		PCD Cooperative fit function
			data: data to be fitted (events x features)
			target: class labels - sparse targets (events x number of classes)

			train_ids:  train indexes - user generated
			test_ids: test indexes - user generated
			trn_params: class TrnParams (optional)
		"""		
		print 'PCD Cooperative fit function'

		if trn_params is None:
			trn_params = TrnParams()
		
		#print 'Train Parameters'	
		#trn_params.Print()
		
		if trn_params.verbose: 
			print 'PCD Cooperative Model Struct: %i - %i - %i'%(data.shape[1],1,target.shape[1])

		for ipcd in range(self.n_components):
			print 'Training %i PCD of %i PCDs'%(ipcd+1,self.n_components)
			
			if ipcd == 0:
				my_model = Sequential()
				
				# add a linear layer to isolate the input of NN model
				my_model.add(Dense(data.shape[1],input_dim=data.shape[1], init='identity',trainable=False))
				
				my_model.add(Activation('linear'))
				
				# add a non-linear single neuron layer to compress all information
				my_model.add(Dense(1, input_dim=data.shape[1], init='uniform'))
				my_model.add(Activation('tanh'))
				
				# add a non-linear output layer with max sparse target shape
				my_model.add(Dense(target.shape[1], init='uniform')) 
				my_model.add(Activation('tanh'))
				
				# creating a optimization function using steepest gradient
				sgd = SGD(lr=trn_params.learning_rate, decay=trn_params.learning_decay, 
                          		  momentum=trn_params.momentum, nesterov=trn_params.nesterov)
				
				# compile the model
				my_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy','mean_squared_error'])
				
				# early stopping to avoid overtraining
				earlyStopping = callbacks.EarlyStopping(monitor='val_loss', 
                                                        patience=25, verbose=0, 
                                                        mode='auto')
				
				# Train model
				init_trn_desc = my_model.fit(data[train_ids], target[train_ids],
                                          		  nb_epoch=trn_params.n_epochs, 
                                          		  batch_size=trn_params.batch_size,
                                          		  callbacks=[earlyStopping], 
                                          		  verbose=trn_params.train_verbose,
                                          		  validation_data=(data[test_ids],
                                                          		   target[test_ids]),
                                          		  shuffle=True)
				self.models[ipcd] = my_model
				self.trn_descs[ipcd] = init_trn_desc
                		self.pcds[ipcd] = my_model.layers[2].get_weights()[0]
				if trn_params.verbose : 
					print 'PCD %i - Train process is done, val_cost: %1.5f'%(ipcd+1,np.min(init_trn_desc.history['val_loss']))
			else:
				my_model = Sequential()
				
				#  I removed the linear layer to allow freeze!!!				
				# add a non-linear freeze previous extracted pcd
				freeze_layer = Sequential()
				freeze_layer.add(Dense(ipcd, input_dim=data.shape[1],trainable=False))
				weights = freeze_layer.layers[0].get_weights()
				
				# loop over previous pcds
				for i_old_pcd in range(ipcd):
					for idim in range(data.shape[1]):
						weights[0][idim,i_old_pcd] = self.pcds[i_old_pcd][idim]

				freeze_layer.layers[0].set_weights(weights)

				# add a non-linear no-freeze new neuron
				non_freeze_layer = Sequential()
				non_freeze_layer.add(Dense(1, input_dim=data.shape[1]))

				# merge everything
				merged = Merge([freeze_layer, non_freeze_layer], mode='concat')
				my_model.add(merged)
				my_model.add(Activation('tanh'))				

				# add a non-linear output layer with max sparse target shape
				my_model.add(Dense(target.shape[1], init='uniform')) 
				my_model.add(Activation('tanh'))
			
				# creating a optimization function using steepest gradient
				sgd = SGD(lr=trn_params.learning_rate, decay=trn_params.learning_decay, 
                          		  momentum=trn_params.momentum, nesterov=trn_params.nesterov)
				
				# compile the model
				my_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy','mean_squared_error'])
				
				# early stopping to avoid overtraining
				earlyStopping = callbacks.EarlyStopping(monitor='val_loss', 
                                                        patience=25, verbose=0, 
                                                        mode='auto')
				# Train model
				init_trn_desc = my_model.fit([data[train_ids], data[train_ids]], target[train_ids],
                                          		  nb_epoch=trn_params.n_epochs, 
                                          		  batch_size=trn_params.batch_size,
                                          		  callbacks=[earlyStopping], 
                                          		  verbose=trn_params.train_verbose,
                                          		  validation_data=([data[train_ids], data[train_ids]],
                                                          		   target[test_ids]),
                                          		  shuffle=True)
				self.models[ipcd] = my_model
				self.trn_descs[ipcd] = init_trn_desc
                		self.pcds[ipcd] = my_model.layers[0].layers[1].get_weights()[0]
				if trn_params.verbose:
					print 'PCD %i - Train process is done, val_cost: %1.5f'%(ipcd+1,np.min(init_trn_desc.history['val_loss']))
		return self

	def transform_with_activation_function(self, data):
		"""
			PCD Cooperative auxiliar transform function
				data: data to be transformed (events x features)
		"""
		output = []
		
		if self.n_components == 1:
			# get the output of an intermediate layer
			# with a Sequential model
			
			get_layer_output = K.function([self.models[self.n_components-1].layers[0].input],[self.models[self.n_components-1].layers[2].output])
			output = get_layer_output([data])[0]
		else:
			print "TO DO!!!" 
			return 0
		return output

	def transform_without_activation_function(self, data):
		"""
			PCD Cooperative auxiliar transform function
				data: data to be transformed (events x features)
		"""

		output = []
		if self.n_components == 1:
			# get the output of an intermediate layer
			# with a Sequential model
			
			get_layer_output = K.function([self.models[self.n_components-1].layers[0].input],[self.models[self.n_components-1].layers[3].output])
			output = get_layer_output([data])[0]
		else:
			for ipcd in range(self.n_components):
				pcd_output = (np.inner(self.pcds[ipcd].T,data).T/np.inner(self.pcds[ipcd].T,self.pcds[ipcd].T))
				if ipcd == 0:
					output = pcd_output
				else:
					output = np.append(output,pcd_output,axis=1)

		return output

	def transform(self, data, use_activation=False):
		"""
			PCD Cooperative transform function
				data: data to be transformed (events x features)
				use_activation: boolean to use or not the activation function
		"""
		if use_activation:
			return self.transform_with_activation_function(data)
		else:
			return self.transform_without_activation_function(data)

	def get_degree_matrix(self):
		degree_matrix = np.zeros([self.n_components,self.n_components])

		if self.models == {}:
			return degree_matrix
		
		for ipcd in range(self.n_components):
			for jpcd in range(self.n_components):
				degree = (np.inner(self.pcds[ipcd].T,self.pcds[jpcd].T)/
					 (np.linalg.norm(self.pcds[ipcd])*np.linalg.norm(self.pcds[jpcd])))
				degree = round(degree.real,6)
				degree = np.arccos(degree)
				degree = np.degrees(degree)
				if degree > 90 and degree < 180:
					degree = degree - 180
				if degree > 180 and degree < 270:
					degree = degree - 180
				degree_matrix[ipcd,jpcd] = np.abs(degree)
		return degree_matrix
