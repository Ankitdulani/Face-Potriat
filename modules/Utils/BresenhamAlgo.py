import operator
import numpy as np
from modules.Geometry import Point
from modules.Utils import ColorSchema

class bresenhamLine:
	def __init__ (self):
		print ("context initiated")

	## Takes input in Point2D format
	## Also a list of tuples
	def drawLine(img,colorPlateValue=0, *argv):

		x= ColorSchema.ColorPlate(colorPlateValue)
		colorValue = x.foregroundColor


		vertexes =[] 

		isColor = False
		if len(img.shape) > 2:
			isColor = True

		if len(argv) > 2:
			return img

		elif len(argv) == 1:
			vertexes=argv[0]

		else :
			vertexes=[(int(argv[0].x),int(argv[0].y)),(int(argv[1].x),int(argv[1].y))]


		######## Follow the Link for derivation
		######## https://www.tutorialspoint.com/computer_graphics/line_generation_algorithm.html

		#Sort the List on the basis of X cordinate
		vertexes = sorted ( vertexes, key=operator.itemgetter(0))
		x1 = int(vertexes[0][0])
		y1 = int(vertexes[0][1])
		x2 = int(vertexes[1][0])
		y2 = int(vertexes[1][1])

		## Flag to hold is slope negative 
		slopenegative = True if (y2-y1) < 0 else False

		dx = abs(x2 -x1)
		dy = abs(y2 -y1)
		a= dx
		b= dy

		# Flag hold the value whether  slope is less than 1
		flag = True
		if dy > dx:
			a = dy
			b = dx
			flag =False

		###### Getting the list of all coordinates
		p = 2*b - a
		count = 0

		for i in range (a+1):

			### Slope less than 1
			if flag == True :
				img[ y1 + count][x1+i] = np.uint8(0) if isColor == False else colorValue#np.zeros(3 , dtype = np.uint8)

			### slope greater than 1
			else:
				#### slope negative check 
				if slopenegative == False:
					img [y2 - i][x2 - count] = np.uint8(0) if isColor == False else colorValue#np.zeros(3 , dtype = np.uint8)
					# print(img [y2 - i][x2 - count])
				else:
					img [y2 +i ][x2 - count] = np.uint8(0) if isColor == False else colorValue#np.zeros(3 , dtype = np.uint8)


			if p >=0:
				count += int( -1 ) if slopenegative == True else int(1)
				p += (2*b - 2*a)

			else:
				p += (2*b)

		return img
