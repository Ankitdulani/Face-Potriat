import numpy as np 
from modules.Utils import IOImage
import cv2

class IP:
	def __init__ (self):
		print (" image Processing initiated")

	def addIntensityMap(img , intensityMap):

		newImg = np.zeros(img.shape, dtype= np.uint8)
		intensityMap = cv2.GaussianBlur(intensityMap, ksize=(15, 15),sigmaX=0, sigmaY=0)

		I =0.0
		maxiIntensityAllowed = 0.99
		scaleI = 1#((maxiIntensityAllowed-I)/(maxI - minI))

		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				colorValue = img[i][j]
				Intensity = I + intensityMap[i][j]*scaleI
				newImg[i][j] = [np.uint8(abs(colorValue[0]*Intensity)),np.uint8(abs(colorValue[1]*Intensity)),np.uint8(abs(colorValue[2]*Intensity))]

		return newImg

	def adjust_gamma(image, gamma=1.3):

		print(gamma)
		table  = tableI = np.array([(((i)/ 255.0) ** gamma) * 255
			for i in np.arange(0, 256)]).astype("uint8")

		invGamma = 1.0 / gamma

		tableI = np.array([(((i)/ 255.0) ** invGamma) * 255
			for i in np.arange(0, 256)]).astype("uint8")

		l , b = image.shape[:2]
		base = 60

		for i in range(l):
			for j in range(b):
				x = image[i][j]
				image [i][j] = table[x] if x < base else tableI[x]
		
		return image
		return cv2.LUT(image, tableI).astype('uint8')

	def createBlackWhite(image, gamma = 5):

		if len( image.shape) > 2:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# image = cv2.GaussianBlur(image, ksize=(11,11),sigmaX=0, sigmaY=0)
		image = IP.adjust_gamma(image,gamma)

		return image


	def createSketch(img_gray, grudge ,Writer):

		#Converting a Colored to GrayScale
		if len( img_gray.shape) > 2:
			img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

		if len( grudge.shape) > 2:
			grudge = cv2.cvtColor(grudge, cv2.COLOR_BGR2GRAY)

		img_gray = cv2.resize( img_gray , (512,512) )
		grudge = cv2.resize( grudge , img_gray.shape)
		
		img_gray_inv = 255 - img_gray

		# This step make the darker point lighter
		img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(51, 51),sigmaX=0, sigmaY=0)

		#This brings out a image with darker points
		img_blend = cv2.divide(img_gray, 255-img_blur, scale=256)

		# Erosion to reduce noise from the image 
		kernel = np.ones((5,5),np.uint8)
		erosion = cv2.erode(255-img_blend,kernel,iterations = 1)
		erosion2 = cv2.erode(erosion,kernel,iterations = 1)

		# This Step Darken the boundary
		output = cv2.multiply(img_blend, 255-erosion, scale=1/256)

		# Step Darken the boundary and shift the gradient to saturated one
		factor = 3
		output = cv2.multiply(output, 255-erosion2, scale=1/(256*factor))

		# Linear burn 
		# Lighten the Images
		img_color_dodge = IP.blend_color_dodge(output,img_gray)

		mask_grudge = IP.blend_screen(img_gray,grudge,.2)
		output = IP.blend_overlay(mask_grudge, img_color_dodge,0.85)

		return output

	def adaptiveThreshold (img , threshold =150):

		img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,5)
		return img

	def BlueGreen(img, thres_x,thres_y=0):
		if len( img.shape) > 2:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		result = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
		result.fill(255)
		neagtive_image = img

		result[ 0 : img.shape[0]-thres_y ,  0:img.shape[1]-thres_x , 0] = neagtive_image[ thres_y : img.shape[0] ,thres_x:img.shape[1]]
		result[ img.shape[0]-thres_y : img.shape[0] ,  img.shape[1]-thres_x : img.shape[1], 0] = neagtive_image[ img.shape[0]-thres_y : img.shape[0] ,  img.shape[1]-thres_x : img.shape[1]]
		result[:,:,1] = neagtive_image[:,:]
		result[:,:,2] = neagtive_image[:,:]
		# result[:,threshold:img.shape[1],2] = neagtive_image[:,0:img.shape[1]-threshold]

		return result;

	### Darkening BLENDS
	#The best mode for darkening. Works by multiplying the luminance levels of the current layer’s pixels with the pixels in the layers below. 
	def blend_multiply(A, B,factor=1):
		return cv2.multiply(A,B,scale=(1/(256*factor)))

	#Darker than Multiply, with more highly saturated mid-tones and reduced highlights.
	def blend_color_burn(A,B):
		return 255 - cv2.divide(255-B,A,scale = 256)


	####LIGHTING BLENDS
	# Takes darkes of both and turn them more darker and innverting 
	# thereby increasing the contrast.	
	def blend_screen(A,B,factor):
		return 255 - cv2.multiply(255-A, 255-B, scale=1/(256*factor))

	#Brighter than the Screen blend mode. Results in an intense, contrasty color-typically results in saturated mid-tones and blown highlights.
	def blend_color_dodge(A,B,scale=256):
		return cv2.divide(A, 255-B,scale=scale)

	# Brighter than the Color Dodge blend mode, but less saturated and intense.
	def blend_linear_dodge(A,B):
		return cv2.add(A,B)

	def blend_overlay(A,B,alpha=1):
		return cv2.addWeighted(B, alpha, A, 1 - alpha,0)

	def overlay(img,colorValue):

		# print (colorValue)

		if len( img.shape) > 2:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		n_img = img#(255 - img)
		result = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)

		result[:,:,0] = ((float(colorValue[0])/255)*n_img[:,:])
		result[:,:,1] = ((float(colorValue[1])/255)*n_img[:,:])
		result[:,:,2] = ((float(colorValue[2])/255)*n_img[:,:])

		return result


	def brush(id):

		if id == 0: 
			return np.array(	[[1,0, 1, 1, 0],
								[1, 1, 0, 0, 1],
								[1, 1, 1, 1, 1],
								[0, 1, 1, 1, 1],
								[0, 0, 1, 1, 1]])

             
    




	#def likeInstaPost(img):

		# create a test image with gradient of the object which is need to be dispalyed generally neon color
		# then use the vivid light concept as the blend mode
		# Descibing vivid light
			# if the color is lighter than 50% gray lightned it
			# if the color is darker than 50% gray darkened it
		# also the intesity of the light also decays away


		# add the object at the centric location

		# Features that can we added
			# Remove the background from the image and replace it by a darker one,
			# also detect the face of the human and take a decision where to add the object
			# Adjust the lighting effect of the image
			# depending upon the source of light color also changes.

		# ADVANCED PROJECT
		# add on feature can make any thing neon 













