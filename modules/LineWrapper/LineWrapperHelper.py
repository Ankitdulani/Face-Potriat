from modules.Utils import IOImage
from modules.Utils import Voxeliser
from modules.Geometry import Point
from modules.Geometry import Line
from modules.Geometry import Surface


import numpy as np
import math 
import random

class Helper:

	__instance = None

	@staticmethod 
	def getInstance():
		if Helper.__instance == None:
			Helper()
		return Helper.__instance

	def __init__(self):
		if Helper.__instance != None:
			raise Exception("This class is a singleton!")
		else:
			Helper.__instance = self

	def computeLine(self,displacementMap,seed,edge_detector):

		scale = 0.25
		power_2 = []
		power_2.append(1);
		for i in range(12):
			power_2.append(power_2[i]*2)

		currY = seed
		prevY = currY

		prevX = 0
		currX = 0

		prevZ = None
		currZ = None

		left = 0.0

		(l , b)  =  displacementMap.shape
		coordinateY=np.zeros( b ,dtype = int)
		scaleLeft = 1

		maxalpha = 0

		while currX < b:
			
			currZ = int(displacementMap[currY][currX])

			if currX == 0:
				prevX = currX
				prevY = currY
				prevZ = currZ
				
			coordinateY[currX]= currY
			
			if currX == 0:
				currX+=1
				continue


			### dampler_inverse ki  
			damper = 0.5
			prevY = currY 
			alpha = scale*(currZ-prevZ) + scaleLeft*left

			maxalpha = max(maxalpha,abs(alpha))
			roundAlpha = int (round(alpha,0))
			
			while abs(roundAlpha) > 3:
				alpha = roundAlpha*damper
				roundAlpha = int (round(alpha,0))


			if (abs(roundAlpha) == 1 and currY <l-1 and currX<b-1):
				if not(edge_detector[currY][currX]>0 and (int(edge_detector[currY+1][currX+1]) +edge_detector[currY-1][currX+1])==0 and edge_detector[currY][currX+1]>0):
					currY = currY + roundAlpha
			
			# currY = currY + roundAlpha
			prevX = currX
			currX += 1


			left = alpha - roundAlpha
			prevZ = currZ
		
		# print(maxalpha)

		return coordinateY

	# For Returns Disparity from displacementMap 
	def getDisparityImage( self,displacementMap):

		img = np.zeros((displacementMap.shape[0],displacementMap.shape[1]) ,dtype = np.uint8)
		img.fill(255)
		zmax=np.amax(displacementMap)
		# print("zmax in displacementMap",zmax)

		maxI = 200
		minI = 50

		for i in range(displacementMap.shape[0]):
			for j in range(displacementMap.shape[1]):
				if displacementMap[i][j] > 0:
					img[i][j] = minI +int ( float ((maxI-minI)*displacementMap[i][j])/float(1*zmax))
		return img

	# Return Random list of y corrdinated for lines
	def getRandomLinesList( self,len,start=1,maxInterval=10,maxWidth = 5,minWidth = 3):

		print("Generating Random Lines ", minWidth, maxWidth, maxInterval)	
		randomLines=[]
		x = start
		minInterval = 3

		while x < len + start - maxWidth:
			width = random.randrange(minWidth,maxWidth,1)
			randomLines.append(x)
			randomLines.append(x+width)
			x += width
			x+= random.randrange(minInterval,maxInterval,1)

		return randomLines

	# outputs the list of distroted value for waverlength  and amplitude
	def getDistrotion(self, wavelength = 20, amplitude = 7, distrotionType = 'sin' ):
		distrotion = np.zeros(wavelength,dtype=int)
		for i in range(wavelength):
			if distrotionType == 'sin':
				distrotion[i] = int(amplitude * math.sin( math.pi * (float(2*i)/float(wavelength))))
			elif distrotionType == 'randomised':
				distrotion[i] = random.randrange(0, amplitude+1 , 1)

		return distrotion

	# Based upon the Key feature it output the value iterative 
	def drawSmoothLine(self,points, itr = 10 , ratio =9 ):

		new_points =[]
		l = len(points)
		while(itr > 0):
			for i in range(l):
				if i ==0 or i == l-1:
					new_points.append(points[i])
					continue

				ptA = Point.Point2D()
				ptA.x = (points[i-1].x + ratio*(points[i].x))/(ratio+1)
				ptA.y = (points[i-1].y + ratio*(points[i].y))/(ratio+1)

				ptB = Point.Point2D()
				ptB.x = (points[i+1].x + ratio*points[i].x)/(ratio+1)
				ptB.y = (points[i+1].y + ratio*points[i].y)/(ratio+1)

				new_points.append(ptA)
				new_points.append(ptB)

			itr -= 1
			points = new_points.copy()
			new_points=[]
			l += (l-2)

		return points

	# Extract important feature and line 
	def extractLineFeature(self,points):

		x_max = len(points)

		threshold = 0.0001
		pt=[]
		ptA = Point.Point2D(0,points[0]) 
		pt.append(ptA)

		for i in range(2,x_max,1):

			ptC = Point.Point2D(i,points[i])
			ptB = Point.Point2D(i-1,points[i-1])

			# get distance
			new_line = Line.Line2D.getLinePoint(ptA, ptC)
			distance = new_line.getDistance(ptB)

			if (distance > threshold):
				pt.append(ptB)
				ptA = ptB

		pt.append(Point.Point2D(x_max-1,points[x_max-1]))
		return pt

	def getSurface(self,extension,name):

		Reader = IOImage.reader()

		if extension == "obj":
			X,Y,Z, Faces,texture = Reader.parseOBJ(name)
			return Surface.surface(X,Y,Z,Faces,texture)

		elif extension == "face":
			#Reading the vertices
			X, Y, Z = Reader.readVertexes(name)
			print ("Number of Vertexs",len(X))

			Faces = (Reader.readFaces(name))
			print ("Number of triangular Faces",len(Faces))

			Texture = Reader.readTexture(name)
			print ("Length of texture File",len(Texture)) 

			## Creating Surface DataType
			return Surface.surface(X,Y,Z,Faces,texture=Texture)

		return None

	def getInputs(self,extension,name,depth=700, baseImage = None):

		Reader = IOImage.reader()
		Writer = IOImage.writer()

		disparityMap = np.array(0)
		colorImage = None
		IntensityMap = np.array(0)
		shadowMap = np.array(0)


		DMFP = "disparityMap_"+name+".txt"
		CIFP = "colorImage_"+name+".jpg"
		IMFP = "intensityMap_"+name+".txt"
		SM = "shadowMap_"+name+".txt"

		if Reader.checkIfExist(DMFP) and Reader.checkIfExist(CIFP) and Reader.checkIfExist(IMFP) and Reader.checkIfExist(SM):
			print("files exists")
			disparityMap = Reader.readNpArray(DMFP)
			colorImage = Reader.readImage(CIFP,img_prefix="",folder="temp",img_suffix="")
			IntensityMap = Reader.readNpArray(IMFP)
			shadowMap = Reader.readNpArray(SM)

		else:
			print("files non exists")
			surface = self.getSurface(extension,name)
			disparityMap, colorImage, IntensityMap,shadowMap = Voxeliser.Voxeliser.getDisparitytMap(surface,h=depth,colorImage=baseImage, margin=1 if extension == "face" else 1.3)
			Writer.writeNpArray(DMFP,disparityMap)
			Writer.writeNpArray(IMFP,IntensityMap)
			Writer.writeNpArray(SM, shadowMap)
			Writer.writeImage(CIFP,colorImage,folder="temp")

		return disparityMap,colorImage,IntensityMap,shadowMap


	