""" 
  This file contents all log functions
"""

import os
import time

import numpy as np
import numpy.random as np_rnd


class DataHandlerFunctions(object):
	def __init__(self):
		self.name = 'DataHandler Class'
        
	def CreateEventsForClass(self, data, n_events, method='reply'):
		print '%s: CreateEventsForClass'%(self.name)
		print 'Original Size: (%i, %i)'%(data.shape[0],data.shape[1])
		if n_events == 0:
			return data
		else:
			if method == 'reply':
				appended_data = data[np_rnd.random_integers(0, data.shape[0], size=n_events),:]
				return_data = np.append(data, appended_data, axis=0)
				return return_data