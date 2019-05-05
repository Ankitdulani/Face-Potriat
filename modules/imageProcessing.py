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

	def addIntensityMap(img , intensityMap):

		maxI = np.amax(intensityMap)
		minI = np.amin(intensityMap)

		newImg = np.zeros(img.shape, dtype= np.uint8)

		I =0.5
		scaleI = ((.99-I)/(maxI - minI))

		print ("Intensity",minI, maxI)

		# print("scalerI", scaleI )

		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				colorValue = img[i][j]
				Intensity = I + intensityMap[i][j]*scaleI
				newImg[i][j] = [np.uint8(abs(colorValue[0]*Intensity)),np.uint8(abs(colorValue[1]*Intensity)),np.uint8(abs(colorValue[2]*Intensity))]
		return newImg