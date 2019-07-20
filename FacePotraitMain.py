import plotly
import plotly.offline as py
import plotly.figure_factory as FF
import plotly.graph_objs as go
import math 
import random
import cv2
import sys
import os.path
from pathlib import Path
import numpy as np
from PIL import Image
from operator import itemgetter
from scipy.spatial import Delaunay
from modules import vector
from modules import Point
from modules import IOImage
from modules import Surface
from modules import Plane
from modules import Voxeliser
from modules import BresenhamAlgo
from modules import Line
from modules import imageProcessing as processor
from modules import Model
from modules import ColorSchema

# Return Random list of y corrdinated for lines
def getRandomLinesList( len,start=1,maxInterval=10,maxWidth = 5,minWidth = 3):

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

# Testig the texture being being copied
def testTexture(surface,l=900,b=1000,h =1000):

	img = np.zeros((l,b,3),dtype = np.uint8)
	disp = np.zeros((l,b),dtype = int)
	img.fill(255)

	X = surface.X
	Y = surface.Y
	Z = surface.Z

	minX = math.floor(min(X)) 
	minY = math.floor(min(Y))
	minZ = math.floor(min(Z))

	maxX = math.ceil(max(X))
	maxY = math.ceil(max(Y))
	maxZ = math.ceil(max(Z))

	minVector = vector.vector3D(minX,minY,minZ)
	#Scale Factor
	ScaleVector = vector.vector3D(int ( b / float(maxX-minX)), int ( l / float(maxY - minY)) , int ( h/(float(maxZ-minZ))))

	# minVector = vector.vector3D(float(-1),float(-1),float(-1))
	# maxZ = max(Z)
	# ScaleVector = vector.vector3D(int ( b / float(2)),int ( l / float(2)),int ( h/(maxZ - float(-1))))

	for i in range( len(X)):
		vec1 = vector.vector3D(X[i],Y[i],Z[i])
		vec1 = vector.vector3D.vectorMultipy(ScaleVector,vector.vector3D.subVector(vec1,minVector))
		pt1 = vec1.vecToIntegerPoint()

		if disp[l-1-pt1.y][pt1.x] < pt1.z:
			disp[l-1-pt1.y][pt1.x] = pt1.z
			img[l-1-pt1.y][pt1.x] = surface.Texture[i]



	return img

# For Creating Testing Example
def createTestDisplacementMap( key, l=900,b=900):	

	displacementMap = np.zeros((l,b) ,dtype = int)
	maxHeight = 300

	## Creates A cube
	if key == "cube":
		displacementMap[300:600,300:600]=int(maxHeight)

	## Creating Displacement Map for Wedge
	if key =="wedge":
		for i in range (300, 600,1):
			displacementMap[i,300:600] = int (maxHeight-(i-int(450)))

	if key == "hemisphere":
		centerY = 450
		centerX = 450
		centerZ = 0
		radius = 150

		# General Equation for sphere is x^2 + y^2 + z^2 =r^2
		for i in range(0, radius +1,1):
			for j in range(0, radius+1,1):

				value = radius**2 - i**2 -j**2

				if value <0:
					continue
				alpha = int (round (math.sqrt(value),0))
				displacementMap[i+centerY][j+centerX]= alpha
				displacementMap[centerY-i][centerX-j]= alpha
				displacementMap[centerY+i][centerX-j]= alpha
				displacementMap[centerY-i][centerX+j]= alpha

	return displacementMap

# For Returns Disparity from displacementMap 
def getDisparityImage( displacementMap):

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

# outputs the list of distroted value for waverlength  and amplitude
def getDistrotion( wavelength = 20, amplitude = 7, distrotionType = 'sin' ):
	distrotion = np.zeros(wavelength,dtype=int)
	for i in range(wavelength):
		if distrotionType == 'sin':
			distrotion[i] = int(amplitude * math.sin( math.pi * (float(2*i)/float(wavelength))))
		elif distrotionType == 'randomised':
			distrotion[i] = random.randrange(0, amplitude+1 , 1)

	return distrotion

# Based upon the Key feature it output the value iterative 
def drawSmoothLine(points, itr = 10 , ratio =9 ):

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
def extractLineFeature(points):

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

# Function which creates the Art
def computeLine(displacementMap,seed,edge_detector):

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
		
		while abs(roundAlpha) > 1:
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

def createArt(displacementMap, img , scale =0.25 , randomLines = [], art_type = 'single_line',colorPlateValue =0, minWidth = 1,maxWidth = 2,maxInterval =5):

	print ("Create Art being called, art_type: ",art_type)
	Writer = IOImage.writer()

	img_prefix="result"
	img_suffix=".jpg"

	foregroundColor=ColorSchema.ColorPlate(colorPlateValue).foregroundColor

	disparityImage = getDisparityImage(displacementMap)
	edge_detector=cv2.Canny(disparityImage,100,150)
	Writer.writeImage('Test_canny.jpg',edge_detector,folder="test")

	isColor = False
	if len(img.shape) >= 3:
		isColor = True

	(l , b)  =  displacementMap.shape
	previousY = np.zeros( b ,dtype = int)

	start = 5

	if len(randomLines) == 0:
		randomLines = getRandomLinesList(len=l-start -1, start= start, maxInterval=maxInterval, maxWidth = maxWidth, minWidth = minWidth)
	print ("length of randomLines",len (randomLines))

	for i in range(len(randomLines)):

		if i % 2 ==0 and art_type =='weighted_line':
			previousY = computeLine(displacementMap, randomLines[i],edge_detector=edge_detector)
			continue

		currY= computeLine(displacementMap, randomLines[i],edge_detector=edge_detector)

		if (art_type == 'smoothen_line'):		
			feature_points = extractLineFeature(currY)
			points = drawSmoothLine(points =feature_points, ratio = 2, itr =6)
			for (index) in range(1,len(points),1):
				img = BresenhamAlgo.bresenhamLine.drawLine(img,colorPlateValue,[(points[index].x,points[index].y),
																(points[index-1].x,points[index-1].y)])
		elif (art_type == 'single_line'):
			for index in range(b):
				# if index % 10 ==0 and i %10 == 0:
				# 	img_name = img_prefix + str.zfill(str(i),5) + str.zfill(str(index),5) +img_suffix
				# 	Writer.writeImage(img_name,img,folder="Continuation")

				img[currY[index]][index] = np.uint8(0) if isColor == False else foregroundColor#np.zeros(3,dtype= np.uint8)

		elif (art_type == 'weighted_line'):
			for index in range(b):
				if (index % 10 ==0 ):
					img_name = img_prefix + str(i) + str(index) +img_suffix
					Writer.writeImage(img_name,img,folder="Continuation")

				img = BresenhamAlgo.bresenhamLine.drawLine(img,colorPlateValue,[(index,currY[index]),(index,previousY[index])])

		elif (art_type == 'sin_line'):
			distrotion = getDistrotion (wavelength = 20 , amplitude = 2)
			for index in range(b-1):
					img[currY[index]+distrotion[(index%20)]][index] = np.uint8(0) if isColor == False else foregroundColor

		elif (art_type == 'randomised_line'):
			distrotion = getDistrotion(wavelength = 20 , amplitude =2,distrotionType='randomised')
			for index in range(b-2):
				img = BresenhamAlgo.bresenhamLine.drawLine(img,colorPlateValue,[(index,currY[index]-(distrotion[index%20])),(index+1,currY[index+1]-(distrotion[(index+1)%20]))])
	return img

# Testing the Intergration part
def Test():

	print ("**** Testing ****")

	r = Model.model.getRotationMatrix(vector.vector3D(0,0,1))
	x = np.array([[0,0,1,0]])
	x= x.transpose()

	print("Rotation matrix testing")

	y = r.dot(x)
	for i in range(y.shape[0]):
		a=[]
		for j in range(y.shape[1]):
			a.append(y[i][j])
		print (a)


	testWriter = IOImage.writer()

	print("testing distrotion")
	points = np.zeros(500)
	points.fill(250)

	wavelength = 50

	distrotion = getDistrotion(wavelength= wavelength, amplitude = 5,distrotionType ='randomised')
	for i in range(500):
		points[i] += distrotion[ i % wavelength ]

	print ("printing extracted points")
	points = extractLineFeature(points)

	print("testing the Line Smoothener")
	# points = [Point.Point2D(100,200),Point.Point2D(300,500),Point.Point2D(200,100),Point.Point2D(500,700)]

	img = np.zeros((900,900) ,dtype = np.uint8)
	img.fill(255)

	for (index) in range(1,len(points),1):
		img = BresenhamAlgo.bresenhamLine.drawLine(img,[((points[index].x),
														 (points[index].y)),
													    ((points[index-1].x),
													     (points[index-1].y))])

	points = drawSmoothLine(points =points, ratio = 3)
	for (index) in range(1,len(points),1):
		img = BresenhamAlgo.bresenhamLine.drawLine(img,[((points[index].x),
														 (points[index].y)),
													    ((points[index-1].x),
													     (points[index-1].y))])

	testWriter.writeImage("testSmoothener_distrotion_featureEc=xtractor.jpg",img,folder="test")



	print( Point.Point3D.getCentroid([Point.Point3D(1,2,3),Point.Point3D(2,3,4),Point.Point3D(3,4,5)]))
	normal = vector.vector3D(0,1,0)
	incidentRay = vector.vector3D(3,-4,0)

	reflected = vector.vector3D.getReflectedVector(normal , incidentRay)

	reflected.printVector()

	# testing Voxelisation 
	X = [-0.2 , 0.0 , 0.2] 
	Y = [-0.2, 0.4 , -0.2]
	Z = [0.0 , 0.0 , 0.0]

	face = [[ 0 ,1 ,2]]
	texture = [[200,0 ,0]]

	surface = Surface.surface(X,Y,Z,face,texture)

	disparityMap, colorImage, IntensityMap = Voxeliser.Voxeliser.getDisparitytMap(surface)

	testWriter.writeImage('TestPotrait.png',colorImage,folder="test")

	
	ptA = Point.Point3D(4,0,0)
	ptB = Point.Point3D(0,4,0)
	ptC = Point.Point3D(0,0,4)

	newPlane = Plane.Plane3D.getPlaneEquation( ptA , ptB , ptC)
	newPlane.printEquation()

	## Test the vector2D 
	A= vector.vector2D(3,0)
	mod = A.getModulus()
	A = A.unitVector()
	A = A.scaledVector(5)
	A = A.addVector(vector.vector2D(0,3))
	A.printVector()
	print( mod)

	## Testing the vector 3D
	B = vector.vector3D(3,2,0)
	B.printVector()

	###### Testing bresenham Line
	img = np.zeros((900,900) ,dtype = np.uint8)
	img.fill(255)
	Art = BresenhamAlgo.bresenhamLine.drawLine(img,Point.Point2D(300,700),Point.Point3D(200,600))
	testWriter.writeImage('bresenhamLine_1.png',Art,folder="test")

	#### Testing the create Art Algorithm

	print ("Testing on hemisphere")
	displacementMap = createTestDisplacementMap(key ="hemisphere")
	img = np.zeros((900,900) ,dtype = np.uint8)
	img.fill(255)


	Art = createArt(displacementMap=displacementMap,  img = img , scale =0.25 , art_type = 'single_line')
	testWriter.writeImage('createArtTestOutput_1.png',Art,folder="test")


	print("testing Text wrapping")
	reader = IOImage.reader()
	img = reader.readImage("ankit_text")
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	disparityMap = cv2.resize(img, (1000,1000))
	disparityMap = 255 - disparityMap


	baseImage = np.zeros((1000,1000), dtype=np.uint8)
	baseImage.fill(255)

	print (disparityMap.shape)
	print (img.shape)

	Writer = IOImage.writer()

	randomLines = getRandomLinesList(800)

	Art = createArt(displacementMap = disparityMap, img = baseImage, randomLines = randomLines ,art_type='single_line')
	Writer.writeImage('text_test.png',Art,folder="test")


	print("Testing the parseOBJ")
	Reader= IOImage.reader()

	#Lamborghini_Aventador, Jess_Casual_Walking_001
	X,Y,Z, Faces,texture = Reader.parseOBJ("malebody")
	surface = Surface.surface(X,Y,Z,Faces,texture)

	Writer = IOImage.writer()
	disparityMap, colorImage, LightMap=Voxeliser.Voxeliser.getDisparitytMap(surface=surface,h=700)
	img = getDisparityImage(disparityMap)
	Writer.writeImage('TestOBJ_parser.jpg',img,folder="test")

	canny =cv2.Canny(img,100,150)
	Writer.writeImage('TestOBJ_canny.jpg',canny,folder="test")

	if texture != None:
		img = testTexture(surface)
		Writer.writeImage('TestTextureOBJ.jpg',img,folder="test")

	Art =  createArt(disparityMap,img,art_type='single_line')
	Writer.writeImage('TestARTOBJ1.jpg',Art,folder="test")



	print ("testing Over")

def getSurface(extension,name):

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

def getInputs(extension,name,depth=700, baseImage = None):

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
		surface = getSurface(extension,name)
		disparityMap, colorImage, IntensityMap,shadowMap = Voxeliser.Voxeliser.getDisparitytMap(surface,h=depth,colorImage=baseImage, margin=1 if extension == "face" else 1.3)
		Writer.writeNpArray(DMFP,disparityMap)
		Writer.writeNpArray(IMFP,IntensityMap)
		Writer.writeNpArray(SM, shadowMap)
		Writer.writeImage(CIFP,colorImage,folder="temp")

	return disparityMap,colorImage,IntensityMap,shadowMap

def create_art_text(name,art_type,minWidth,maxWidth,maxInterval,colorPlate):

	reader = IOImage.reader()
	Writer = IOImage.writer()

	img = reader.readImage(name,img_prefix="",folder="Text")
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	disparityMap = cv2.resize(img, (1000,1000))
	disparityMap = 255 - disparityMap

	disparityMap=processor.IP.createBlackWhite(disparityMap)
	Writer.writeImage("test.jpg",disparityMap,folder="Text")

	img = cv2.Canny(disparityMap,10,80)
	baseImage = 255-img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# baseImage = np.zeros((1000,1000), dtype=np.uint8)
	# baseImage.fill(255)

	fileName = "Art_"+str(name)+".jpg"
	Art = createArt(displacementMap = disparityMap, img = baseImage, art_type=art_type,maxInterval=int(maxInterval),colorPlateValue=colorPlate, maxWidth = int(maxWidth), minWidth = int(minWidth))
	Writer.writeImage(fileName,Art,folder="Text")

def create_art_obj(name,art_type,minWidth,maxWidth,maxInterval,colorPlate,background = "canny"):

	print("Creating Art of OBJ files")
	Writer = IOImage.writer()

	# print(colorPlate)
	disparityMap, colorImage, LightMap , shadowMap = getInputs("obj",name)

	img = getDisparityImage(shadowMap)
	Writer.writeImage("shadowMap.jpg",img,folder=name)


	baseImage = np.zeros(colorImage.shape,dtype=np.uint8)
	baseImage.fill(255)
	if background == "Dis":
		baseImage = getDisparityImage(disparityMap)
		Writer.writeImage("baseImage.jpg",baseImage,folder=name)

	if background == "color":
		baseImage = colorImage

	if background =="canny":
		print("Using canny")
		disparityImage = getDisparityImage(disparityMap)
		img = cv2.Canny(disparityImage,10,80)
		baseImage = 255-img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	baseImage = processor.IP.overlay(baseImage,ColorSchema.ColorPlate(colorPlate).backgroundColor)
	Writer.writeImage("baseImage_canny.jpg",baseImage,folder=name)

	# colorImageLight = processor.IP.addIntensityMap(colorImage, LightMap)
	# Writer.writeImage('color_intensity.png',colorImageLight,folder=name)

	Art =  createArt(disparityMap,baseImage,art_type=art_type,maxInterval=int(maxInterval),colorPlateValue=colorPlate, maxWidth = int(maxWidth), minWidth = int(minWidth))
	Writer.writeImage("Art_OBJ.jpg",Art,folder=name)

	shifted_image = processor.IP.BlueGreen(Art,10,5)
	Writer.writeImage('Shifted_Hue.png',shifted_image,folder=name)

def create_art_potrait(name,art_type,depth,minWidth,maxWidth,maxInterval,colorPlate):

	Reader = IOImage.reader()
	Writer = IOImage.writer()

	DisplaySurface = bool(True)

	img = Reader.readImage(name)
	resized_image = cv2.resize(img,(1000,1000))

	### test the tesxture
	# Art = testTexture(surface)
	# fileName = "Test_Testure"+name+".jpg"
	# Writer.writeImage('test_texture.png',Art,folder=name)

	#get the all possible Edges
	# Edges = surface.getEdges()
	# print ("Total Number of Edges",len(Edges))

	# boundaryEdges= surface.getBoundaryEdges()
	# print ("Total Number of Boundary Edges",len(boundaryEdges)) 

	# surface.getExtendedSurfaceToBase( boundaryEdges)
	# print("Total number of Faces surface 1 ",len(surface.Faces))

	disparityMap, colorImage, IntensityMap ,shadowMap= getInputs("face",name,depth=depth,baseImage=None)

	# colorImageLight = processor.IP.addIntensityMap(colorImage, IntensityMap)
	# Writer.writeImage('ColorWithIntensity_1_0.png',colorImageLight,folder=name)

	# img123 = cv2.GaussianBlur(colorImageLight,(9,9),0)
	# Writer.writeImage('ColorWithIntensityLight_Blur_1_0.png',img123,folder=name)
	
	# grayScaleImage = cv2.cvtColor(img123, cv2.COLOR_BGR2GRAY)
	# Writer.writeImage('GrayWithIntensityLight_Blur_1_0.png',grayScaleImage,folder=name)

	# grayScaleImage1 = cv2.cvtColor(colorImageLight, cv2.COLOR_BGR2GRAY)
	# Writer.writeImage('GrayWithIntensityLight_1_0.png',grayScaleImage1,folder=name)

	# grayScaleImage2 = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
	# Writer.writeImage('Gray_1_0.png',grayScaleImage2,folder=name)

	# Sketch = processor.IP.createSketch(colorImage)
	# Writer.writeImage('SketchColor_1_0.png',Sketch,folder=name)

	# Sketch2 = processor.IP.createSketch(img123)
	# Writer.writeImage('SketchColorWithLight_1_1.png',Sketc,folder=nameh)

	# # Verifying the Voxelisation visually
	# Art=getDisparityImage(disparityMap)
	# Writer.writeImage('DisplacemntMapFace_1_0.png',Art,folder=name)

	adjusted = processor.IP.createBlackWhite(image = resized_image, gamma = 1.0)
	Writer.writeImage('Adjusted_Image.png',adjusted,folder=name)

	ColorBase = processor.IP.overlay(adjusted,ColorSchema.ColorPlate(colorPlate).backgroundColor)
	Writer.writeImage('Background_Image.png',ColorBase,folder=name)

	shifted_image = processor.IP.BlueGreen(adjusted,15)
	Writer.writeImage('Shifted_Hue.png',shifted_image,folder=name)


	(l,b) = disparityMap.shape
	start = 1
	randomLines = getRandomLinesList(len=l-start -1, start= start,  maxInterval=int(maxInterval), maxWidth = int(maxWidth), minWidth = int(minWidth))

	# ### Creating the Art
	# Art = createArt(displacementMap = disparityMap, img = colorImage , randomLines = randomLines)
	# Writer.writeImage('FacePotrait_1_0.png',Art,folder=name)

	# Art = createArt(displacementMap = disparity,Map, img = colorImageLight, randomLines = randomLines )
	# Writer.writeImage('FacePotrait_1_1.png',Art,folder=name)

	# Art = createArt(displacementMap = disparityMap, img = img123 , randomLines = randomLines)
	# Writer.writeImage('FacePotrait_1_2.png',Art,)

	# Art = createArt(displacementMap = disparityMap, img = grayScaleImage , randomLines = randomLines)
	# Writer.writeImage('FacePotrait_1_3.png',Art,folder=name)

	# Art = createArt(displacementMap = disparityMap, img = grayScaleImage1 , randomLines = randomLines)
	# Writer.writeImage('FacePotrait_1_4.png',Art,folder=name)

	# Art = createArt(displacementMap = disparityMap, img = grayScaleImage2 , randomLines = randomLines)
	# Writer.writeImage('FacePotrait_1_5.png',Art,folder=name)

	# Art = createArt(displacementMap = disparityMap, img = Sketch , randomLines = randomLines)
	# Writer.writeImage('FacePotrait_1_6.png',Art,folder=name)

	Art = createArt(displacementMap = disparityMap, img = adjusted , randomLines = randomLines,art_type=art_type)
	Writer.writeImage('FacePotrait_adjusted.png',Art,folder=name)


	Art = createArt(displacementMap = disparityMap, img = shifted_image, randomLines = randomLines,art_type=art_type)
	Writer.writeImage('FacePotrait_shifted.png',Art,folder=name)

	Art = createArt(displacementMap = disparityMap, img = ColorBase, randomLines = randomLines,art_type=art_type,colorPlateValue=colorPlate)
	Writer.writeImage('FacePotrait_colored.png',Art,folder=name)

	if DisplaySurface == False:

		fig1 = FF.create_trisurf(x=surface.X, y=surface.Y, z=surface.Z,
									# color_func = surface.Texture,
									colormap = [(0.4, 0.15, 0), (1, 0.65, 0.12)],
									simplices=surface.Faces,
									title="Mobius Band",
									plot_edges=False,
									show_colorbar=False
									)


		py.plot(fig1, filename="face.html")

if __name__ == '__main__':

	# Test()

	#Lamborghini_Aventador, Jess_Casual_Walking_001
	name = sys.argv[1]
	extension = sys.argv[2]
	art_type = sys.argv[3] if len(sys.argv) > 3 else 'single_line'
	minWidth = sys.argv[4] if len(sys.argv) > 4 else 1
	maxWidth = sys.argv[5] if len(sys.argv) > 5 else 3
	maxInterval = sys.argv[6] if len(sys.argv) > 6 else 10
	depth =sys.argv[7] if len(sys.argv) >  7 else 700
	colorPlates = sys.argv[8] if len(sys.argv) > 8 else 0



	if (str(extension) == "obj"):
		create_art_obj(name,art_type,minWidth,maxWidth,maxInterval,colorPlates)
	elif (str(extension) == "text"):
		create_art_text(name,art_type,minWidth,maxWidth,maxInterval,colorPlates)
	else :
		create_art_potrait(name,art_type,depth,minWidth,maxWidth,maxInterval,colorPlates)


