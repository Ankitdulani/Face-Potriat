from modules.Utils import IOImage
from modules.Utils import BresenhamAlgo
from modules.Utils import ColorSchema
from modules.LineWrapper import LineWrapperHelper
from modules.LineWrapper import LineWrapperHelper
import cv2
import numpy as np

class LineWrapperImpl:

	def __init__(self):

		self.Writer= IOImage.writer()
		self.img_prefix="result"
		self.img_suffix=".jpg"
		self.helper = LineWrapperHelper.Helper.getInstance(); 

	def isColorImage(self,img):
		if len(img.shape) >= 3:
			return True;
		return False

	# Rerturn Image of Art Created
	def draw(self,displacementMap, img , scale =0.25 , randomLines = [], art_type = 'single_line',colorPlateValue =0, minWidth = 1,maxWidth = 2,maxInterval =5):

		print ("Creating Art, art_type: ",art_type)
		foregroundColor=ColorSchema.ColorPlate(colorPlateValue).foregroundColor

		disparityImage = self.helper.getDisparityImage(displacementMap)
		edge_detector=cv2.Canny(disparityImage,100,150)
		self.Writer.writeImage('Test_canny.jpg',edge_detector,folder="test")

		isColor = self.isColorImage(img)

		(l , b)  =  displacementMap.shape
		previousY = np.zeros( b ,dtype = int)

		start = 5

		if len(randomLines) == 0:
			randomLines = self.helper.getRandomLinesList(len=l-start -1,
														 start= start,
														 maxInterval=maxInterval,
														 maxWidth = maxWidth,
														 minWidth = minWidth)
		print ("Length of randomLines",len (randomLines))
		for i in range(len(randomLines)):

			if i % 2 ==0 and art_type =='weighted_line':
				previousY = self.helper.computeLine(displacementMap,
													randomLines[i],
													edge_detector=edge_detector)
				continue

			currY=  self.helper.computeLine(displacementMap,
											randomLines[i],
											edge_detector=edge_detector)
			if (art_type == 'single_line'):
				for index in range(b):
					# if index % 10 ==0 and i %10 == 0:
					# 	img_name = img_prefix + str.zfill(str(i),5) + str.zfill(str(index),5) +img_suffix
					# 	Writer.writeImage(img_name,img,folder="Continuation")
					img[currY[index]][index] = np.uint8(0) if isColor == False else foregroundColor#np.zeros(3,dtype= np.uint8)

			elif (art_type == 'weighted_line'):
				for index in range(b):
					#if (index % 10 ==0 ):
						#img_name = img_prefix + str(i) + str(index) +img_suffix
						#Writer.writeImage(img_name,img,folder="Continuation")
					img = BresenhamAlgo.bresenhamLine.drawLine(img,
																colorPlateValue,
																[(index,currY[index]),(index,previousY[index])])

			elif (art_type == 'smoothen_line'):		
				feature_points = self.helper.extractLineFeature(currY)
				points = self.helper.drawSmoothLine(points =feature_points, ratio = 6, itr =10)

				for (index) in range(1,len(points),1):
					img = BresenhamAlgo.bresenhamLine.drawLine(img,
																colorPlateValue,
																[(points[index].x,points[index].y),(points[index-1].x,points[index-1].y)])
			

			elif (art_type == 'sin_line'):
				distrotion = self.helper.getDistrotion (wavelength = 20 , amplitude = 2)
				for index in range(b-1):
						img[currY[index]+distrotion[(index%20)]][index] = np.uint8(0) if isColor == False else foregroundColor

			elif (art_type == 'randomised_line'):
				distrotion = self.helper.getDistrotion(wavelength = 20 , amplitude =2,distrotionType='randomised')
				for index in range(b-2):
					img = BresenhamAlgo.bresenhamLine.drawLine(img,
																colorPlateValue,
																[(index,currY[index]-(distrotion[index%20])),(index+1,currY[index+1]-(distrotion[(index+1)%20]))])
		return img