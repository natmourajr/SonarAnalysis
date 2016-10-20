""" 
  This file contents all log functions
"""

import os
import time


class DataHandler(object):
	def __init__(self):
		self.name = 'DataHandler Class'
        
	def CreateEventsForClass(self, data, n_events, method='reply'):
		print '%s: CreateEventsForClass'%(self.name)