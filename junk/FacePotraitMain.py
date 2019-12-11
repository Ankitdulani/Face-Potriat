
import math 
import random
import cv2
import sys
import os.path
from pathlib import Path

from PIL import Image
from operator import itemgetter
from scipy.spatial import Delaunay
from modules import vector

from modules import IOImage

from modules import Plane
from modules import Voxeliser
from modules import BresenhamAlgo

from modules import imageProcessing as processor
from modules import Model
from modules import ColorSchema

from modules import LineWrapper



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
		LineWrapper.Obj.create(name,art_type,minWidth,maxWidth,maxInterval,colorPlates)
	elif (str(extension) == "text"):
		LineWrapper.Text.create(name,art_type,minWidth,maxWidth,maxInterval,colorPlates)
	else :
		LineWrapper.Potrait.create(name,art_type,depth,minWidth,maxWidth,maxInterval,colorPlates)


