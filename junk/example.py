import plotly
import plotly.offline as py
import plotly.figure_factory as FF
import plotly.graph_objs as go
import math 
import random
import cv2
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
from modules import imageProcessing as processor

# Return Random list of y corrdinated for lines
def getRandomLinesList( len,start=1,maxInterval=10):

	randomLines=[]
	x = start
	maxWidth =5 
	minWidth = 1
	while x < len + start - maxWidth:
		width = random.randrange(minWidth,maxWidth,1)
		randomLines.append(x)
		randomLines.append(x+width)
		x += width
		x+= random.randrange(2,maxInterval,1)

	return randomLines

# Testig the texture being being copied
def testTexture(surface,l=900,b=1000,h =1000):

	img = np.zeros((l,b,3),dtype = np.uint8)
	disp = np.zeros((l,b),dtype = int)
	img.fill(255)

	X = surface.X
	Y = surface.Y
	Z = surface.Z

	minVector = vector.vector3D(float(-1),float(-1),float(-1))
	maxZ = max(Z)
	ScaleVector = vector.vector3D(int ( b / float(2)),int ( l / float(2)),int ( h/(maxZ - float(-1))))

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

# For verifying the Disparity Image Visually 
def testDisplacementMap( displacementMap):

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

# For testing Purpose
def makeGray ( img , displacementMap):

	l = int(displacementMap.shape[0])
	b = int(displacementMap.shape[1])

	for i in range ( l ):
		for j in range ( b ):
			img[i][j] = np.uint8(200) if displacementMap[i][j] > 0 else np.uint8(255) 

	return img

# Function which creates the Art
def createArt(displacementMap, img , scale =0.25 , randomLines = [] ):

	print ("Create Art being called")

	isColor = False
	if len(img.shape) >= 3:
		isColor = True

	scaleA = 0.25
	scaleB = 0.2

	scaleLeft = 1

	l = int(displacementMap.shape[0])
	b = int(displacementMap.shape[1])

	# img = testDisplacementMap (displacementMap)
	# img = makeGray (img, displacementMap)

	previousY = np.zeros( b ,dtype = int)

	start = 1
	count = 0

	if len(randomLines) == 0:
		randomLines = getRandomLinesList(len=l-start -1, start= start, maxInterval =15)

	print ("length of randomLines",len (randomLines))

	for i in range(len(randomLines)):

		currY = randomLines[i]
		prevY = currY

		prevX = 0
		currX = 0

		prevZ = None
		currZ = None

		left = 0.0

		if i % 2 ==0:
			previousY.fill(0)
			count +=1

		while currX < b:
			
			currZ = displacementMap[currY][currX]

			if currX == 0:
				img[currY][currX]=np.uint8(0)
				prevX = currX
				prevY = currY
				prevZ = currZ
				currX+=1
				

			if i%2 == 0:
				previousY[currX]=currY
			else:
				img = BresenhamAlgo.bresenhamLine.drawLine(img,[(currX,currY),(currX,previousY[currX])])

			if currX == 0:
				continue

			#create a line prev to current 
			# img[currY][currX]=np.uint8(0)
			
			#### Need modification
			#### Reason it depends upon the slope of boundary 
			##### If slope is < 1 than damper should be adjusted according
			##### Will work afterwards

			damper = 0.5
			prevY = currY 
			alpha = (scale*(currZ-prevZ))  + scaleLeft*left
			roundAlpha = int (round(alpha,0))
			
			while abs(roundAlpha) > 1:
				alpha = roundAlpha*damper
				roundAlpha = int (round(alpha,0))

			left = alpha - roundAlpha

			currY = currY + roundAlpha

			prevX = currX
			currX += 1

			prevZ = currZ

	print("intialised to zero",count)
	return img


# Testing 
def Test():

	print ("**** Testing ****")

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

	testWriter = IOImage.writer()

	testWriter.writeImage('TestPotrait.png',colorImage)

	
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

	##### Display the image

	testWriter.writeImage('bresenhamLine_1.png',Art)
	img = Image.fromarray(Art)
	img.save ('bresenhamLine.png')

	#### Testing the create Art Algorithm
	displacementMap = createTestDisplacementMap(key ="hemisphere")
	
	Art = testDisplacementMap(displacementMap)
	testWriter.writeImage('createArtTestOutput_1.png',Art)

if __name__ == '__main__':

	Test()
	print ("testing Over")


	# This variable intiates html display of surafce using plotly
	DisplaySurface = bool(False)
	backgroundImage = bool(True)
	outputImage = "coloredA"

	Writer = IOImage.writer()

	reader = IOImage.reader()
	# Reading the vertices
	X, Y, Z = reader.readVertexes("vertexSA_new.txt")
	print ("Number of Vertexs",len(X))

	#Reading triangle Vertexs
	Faces = (reader.readFaces("faceSA_new.txt"))
	print ("Number of triangular Faces",len(Faces))

	#Read Texture File
	#Input texture file marks color for each vertex file
	Texture = reader.readTexture("textureSA_new.txt")
	print ("Length of texture File",len(Texture)) 

	## Creating Surface DataType
	surface = Surface.surface(X,Y,Z,Faces,texture=Texture)

	### test the tesxture
	Art = testTexture(surface)
	Writer.writeImage('TextureTestOutput1.png',Art)


	# print(surface.Texture)

	#get the all possible Edges
	# Edges = surface.getEdges()
	# print ("Total Number of Edges",len(Edges))

	# boundaryEdges= surface.getBoundaryEdges()
	# print ("Total Number of Boundary Edges",len(boundaryEdges)) 

	# surface.getExtendedSurfaceToBase( boundaryEdges)
	# print("Total number of Faces surface 1 ",len(surface.Faces))


	disparityMap, colorImage, IntensityMap = Voxeliser.Voxeliser.getDisparitytMap(surface)
	print ("disparity Map Shape",disparityMap.shape)
	Writer.writeImage('ColorImagePotrait_1.png',colorImage)

	
	colorImageLight = processor.IP.addIntensityMap(colorImage, IntensityMap)
	Writer.writeImage('ColorWithIntensity_1_0.png',colorImageLight)

	img123 = cv2.GaussianBlur(colorImageLight,(9,9),0)
	Writer.writeImage('ColorWithIntensityLight_Blur_1_0.png',img123)
	
	grayScaleImage = cv2.cvtColor(img123, cv2.COLOR_BGR2GRAY)
	Writer.writeImage('GrayWithIntensityLight_Blur_1_0.png',grayScaleImage)

	grayScaleImage1 = cv2.cvtColor(colorImageLight, cv2.COLOR_BGR2GRAY)
	Writer.writeImage('GrayWithIntensityLight_1_0.png',grayScaleImage1)

	grayScaleImage2 = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
	Writer.writeImage('Gray_1_0.png',grayScaleImage2)

	Sketch = processor.IP.createSketch(colorImage)
	Writer.writeImage('SketchColor_1_0.png',Sketch)

	Sketch2 = processor.IP.createSketch(img123)
	Writer.writeImage('SketchColorWithLight_1_1.png',Sketch)

	# Verifying the Voxelisation visually
	Art=testDisplacementMap(disparityMap)
	Writer.writeImage('DisplacemntMapFace_1_0.png',Art)

	adjusted = processor.IP.createBlackWhite(image = colorImage, gamma = 2)
	Writer.writeImage('AdjustedGame_1_0.png',adjusted)



	if backgroundImage == False:
		baseImage.fill(255)
	else:
		if outputImage == "colored":
			baseImage = colorImage
		else:
			baseImage = grayScaleImage


	l = int(disparityMap.shape[0])
	b = int(disparityMap.shape[1])
	start = 1

	randomLines = getRandomLinesList(len=l-start -1, start= start, maxInterval =10)

	### Creating the Art
	Art = createArt(displacementMap = disparityMap, img = colorImage , randomLines = randomLines)
	Writer.writeImage('FacePotrait_1_0.png',Art)

	Art = createArt(displacementMap = disparityMap, img = colorImageLight, randomLines = randomLines )
	Writer.writeImage('FacePotrait_1_1.png',Art)

	Art = createArt(displacementMap = disparityMap, img = img123 , randomLines = randomLines)
	Writer.writeImage('FacePotrait_1_2.png',Art)

	Art = createArt(displacementMap = disparityMap, img = grayScaleImage , randomLines = randomLines)
	Writer.writeImage('FacePotrait_1_3.png',Art)

	Art = createArt(displacementMap = disparityMap, img = grayScaleImage1 , randomLines = randomLines)
	Writer.writeImage('FacePotrait_1_4.png',Art)

	Art = createArt(displacementMap = disparityMap, img = grayScaleImage2 , randomLines = randomLines)
	Writer.writeImage('FacePotrait_1_5.png',Art)

	Art = createArt(displacementMap = disparityMap, img = Sketch , randomLines = randomLines)
	Writer.writeImage('FacePotrait_1_6.png',Art)

	Art = createArt(displacementMap = disparityMap, img = adjusted , randomLines = randomLines)
	Writer.writeImage('FacePotrait_1_7.png',Art)


	if DisplaySurface == True:

		fig1 = FF.create_trisurf(x=surface.X, y=surface.Y, z=surface.Z,
									# color_func = surface.Texture,
									colormap = [(0.4, 0.15, 0), (1, 0.65, 0.12)],
									simplices=surface.Faces,
									title="Mobius Band",
									plot_edges=False,
									show_colorbar=False
									)


		py.plot(fig1, filename="face.html")

