import numpy as np 

class IP:
	def __init__ (self):
		print (" image Processing initiated")

	def convertColorToGrayscale( img ):

		l = img.shape[0]
		b = img.shape[1]

		newArray = np.zeros((l,b),dtype = np.uint8 )
		for i in range (l):
			for j in range (b):
				newArray[i][j] = np.uint8(np.mean(img[i,j,:]))

		return newArray