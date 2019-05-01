import plotly
import plotly.offline as py
import plotly.figure_factory as FF
import plotly.graph_objs as go
import math 
import random
import numpy as np
from PIL import Image
from operator import itemgetter
from scipy.spatial import Delaunay
from modules import vector
from modules import Point
from modules import Reader
from modules import Surface
from modules import Plane
from modules import Voxeliser
from modules import BresenhamAlgo


# Return Random list of y corrdinated for lines
def getRandomLinesList( len,start=1,maxInterval=5):

	randomLines=[]
	x = start
	while x < len + start:
		randomLines.append(x)
		x+= random.randrange(0,maxInterval,1)

	return randomLines

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
def createArt(displacementMap, scale =0.25):

	print ("Create Art being called")

	scaleA = 0.25
	scaleB = 0.2

	scaleLeft = 1

	l = int(displacementMap.shape[0])
	b = int(displacementMap.shape[1])

	img = np.zeros((l,b) ,dtype = np.uint8)
	img.fill(255)

	# img = testDisplacementMap (displacementMap)
	# img = makeGray (img, displacementMap)

	start = 1

	randomLines = getRandomLinesList(len=l-start -1, start= start, maxInterval = 6)

	print ("length of randomLines",len (randomLines))

	for i in randomLines:

		currY = i
		prevY = currY

		prevX = 0
		currX = 0

		prevZ = None
		currZ = None

		left = 0.0

		while currX < b:
			
			currZ = displacementMap[currY][currX]

			if currX == 0:
				img[currY][currX]=np.uint8(0)
				prevX = currX
				prevY = currY
				prevZ = currZ
				currX+=1
				continue

			#create a line prev to current 
			# img[currY][currX]=np.uint8(0)
			img = BresenhamAlgo.bresenhamLine.drawLine(img,[(currX,currY),(prevX,prevY)])

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

	
	return img

# Testing 
def Test():

	print ("**** Testing ****")
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
	img = Image.fromarray(Art)
	img.save ('bresenhamLine.png')

	#### Testing the create Art Algorithm
	displacementMap = createTestDisplacementMap(key ="hemisphere")
	
	Art = testDisplacementMap(displacementMap)
	img = Image.fromarray(Art)
	img.save ('createArtTestOutput.png')

if __name__ == '__main__':

	Test()
	print ("testing Over")

	# This variable intiates html display of surafce using plotly
	DisplaySurface = False

	
	reader = Reader.reader()
	# Reading the vertices
	X, Y, Z = reader.readVertexes("vertex_new.txt")
	print ("Number of Vertexs",len(X))

	#Reading triangle Vertexs
	Faces = (reader.readFaces("face_new.txt"))
	print ("Number of triangular Faces",len(Faces))

	## Creating Surface DataType
	surface = Surface.surface(X,Y,Z,Faces)

	#get the all possible Edges
	# Edges = surface.getEdges()
	# print ("Total Number of Edges",len(Edges))

	# boundaryEdges= surface.getBoundaryEdges()
	# print ("Total Number of Boundary Edges",len(boundaryEdges)) 

	# surface.getExtendedSurfaceToBase( boundaryEdges)
	# print("Total number of Faces surface 1 ",len(surface.Faces))


	disparityMap = Voxeliser.Voxeliser.getDisparitytMap(surface)
	print ("disparity Map Shape",disparityMap.shape)

	## Verifying the Voxelisation visually
	Art=testDisplacementMap(disparityMap)
	img = Image.fromarray(Art)
	img.save ('DisplacemntMapFace.png')


	#### Creating the Art
	Art = createArt(displacementMap = disparityMap)

	#### Display the image
	img = Image.fromarray(Art)
	img.save ('FacePotrait.png')


	if DisplaySurface == True:

		fig1 = FF.create_trisurf(x=surface.X, y=surface.Y, z=surface.Z,
									colormap = [(0.4, 0.15, 0), (1, 0.65, 0.12)],
									simplices=surface.Faces,
									title="Mobius Band",
									plot_edges=False,
									show_colorbar=False
									)


		py.plot(fig1, filename="face.html")

