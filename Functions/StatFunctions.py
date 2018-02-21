""" 
  This file contents stat functions
"""

import numpy as np

def KLDiv(p,q):
	if len(p) != len(q):
		print 'Different dimensions'
		return -1

	kl_values = []
	for i in range(len(p)):
		if p[i] == 0 or q[i] == 0 :
			kl_values = np.append(kl_values,0)
		else:
			kl_value = np.absolute(p[i])*np.log10(np.absolute(p[i])/np.absolute(q[i]))
			if np.isnan(kl_value):
				kl_values = np.append(kl_values,0)
			else:
				kl_values = np.append(kl_values,kl_value)
			#print "KLDiv: p= %f, q=%f, kl_div= %f"%(p[i],q[i],kl_values[i])

	return [np.sum(kl_values),kl_values]
