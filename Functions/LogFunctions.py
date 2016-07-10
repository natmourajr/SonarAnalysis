""" 
  This file contents all log functions
"""

import os
import time

class LogInformation(object):
	def __init__(self):
		self.date = None
		self.base = os.environ['OUTPUTDATAPATH']
		
	def CreateLogEntry(self,package_name="NoveltyDetection",analysis_name="SingleClassSVM"):
		# Open the logfile
		if(not os.path.exists(self.base+'/'+package_name)):
			print self.base+'/'+package_name+' doesnt exist...please create it'
			return -1
		
		date = time.strftime("%Y_%m_%d_%H_%M_%S")
		fo = open(self.base+'/'+package_name+'/'+package_name+'_log_file.txt', "a")
		fo.write( "%s %s\n"%(date, analysis_name));
		return date
		
	def RecoverLogEntries(self,package_name="NoveltyDetection"):
		if(not os.path.exists(self.base+'/'+package_name+'/'+package_name+'_log_file.txt')):
			print self.base+'/'+package_name+'/'+package_name+'_log_file.txt'+' doesnt exist...please create it'
			return -1
		fo = open(self.base+'/'+package_name+'/'+package_name+'_log_file.txt', "r")
		log_data = {}
		i = 0
		for line in fo:
			data = {}
			line = line[:-1]
			[a,b] = line.split(" ")
			#print "%s : %s --- %s"%(line,a,b)
			data['date'] = a
			data['package'] = b
			log_data[i] = data
			i = i+1
		return log_data
			
		