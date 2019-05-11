import numpy as np 
import cv2

class IP:
	def __init__ (self):
		print (" image Processing initiated")

	def addIntensityMap(img , intensityMap):

		maxI = np.amax(intensityMap)
		minI = np.amin(intensityMap)

		newImg = np.zeros(img.shape, dtype= np.uint8)
		intensityMap = cv2.GaussianBlur(intensityMap, ksize=(15, 15),sigmaX=0, sigmaY=0)


		I =0.4
		scaleI = ((.99-I)/(maxI - minI))

		print ("Intensity",minI, maxI)

		# print("scalerI", scaleI )

		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				colorValue = img[i][j]
				Intensity = I + intensityMap[i][j]*scaleI
				newImg[i][j] = [np.uint8(abs(colorValue[0]*Intensity)),np.uint8(abs(colorValue[1]*Intensity)),np.uint8(abs(colorValue[2]*Intensity))]
		return newImg

	def adjust_gamma(image, gamma=1.0):

		table  = tableI = np.array([(((i)/ 255.0) ** gamma) * 255
			for i in np.arange(0, 256)]).astype("uint8")

		invGamma = 1.0 / gamma
		tableI = np.array([(((i)/ 255.0) ** invGamma) * 255
			for i in np.arange(0, 256)]).astype("uint8")

		l , b = image.shape[:2]
		base = 120

		for i in range(l):
			for j in range(b):
				x = image[i][j]
				image [i][j] = tableI[x] if x > base else table[x]
		
		return image
		return cv2.LUT(image, tableI).astype('uint8')

	def createBlackWhite(image, gamma = 2):

		if len( image.shape) > 2:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		image = cv2.GaussianBlur(image, ksize=(11,11),sigmaX=0, sigmaY=0)
		image = IP.adjust_gamma(image,gamma)

		return image

	def createSketch(img_gray):

		#Converting a Colored to GrayScale
		if len( img_gray.shape) > 2:
			img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)


		# Creating a Negative Image of it
		img_gray_inv = 255 - img_gray

		# Apply Gausian Blur
		img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),sigmaX=0, sigmaY=0)

		#dogete Negative and the present Image
		img_blend = cv2.divide(img_gray, 255-img_blur, scale=256)

		img_blend = cv2.multiply(img_blend, img_gray, scale=1/256)

		return img_blend

	def adaptiveThreshold (img , threshold =150):

		img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,5)
		return img