import numpy as np

class ColorPlate:

	def __init__ (self,scheme=0):
		self.foregroundColor, self.backgroundColor = ColorPlate.getScheme(int(scheme))

	def getScheme( scheme):
		# 
		if (scheme == 1):
			return  np.array([237, 245, 225]),np.array([5, 56, 107])
		elif (scheme == 0):
			return np.array([0, 0, 0]), np.array([255, 255, 255])
		elif (scheme == 2):
			return np.array([195,7,63]) , np.array([251,238,193])	
			
		elif (scheme == 3):
			return np.array([255,255,255]) , np.array([70, 71, 71])	
		elif (scheme == 4):
			return np.array([251,238,193]) , np.array([195,7,63])	
		if (scheme == 5):
			return  np.array([5, 56, 107]), np.array([237, 245, 225])