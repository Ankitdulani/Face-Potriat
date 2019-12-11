import numpy as np
import math
from modules.Geometry import *
from modules.Utils import vector
from modules.Utils import color
from modules.Utils import Model
from modules.Utils import IOImage
import operator

class Voxeliser:

	def __init__ ( self ):
		print ( "Voxeliser Created" )

	def transform(A, mat):

		t_point = mat.dot(A.transpose())
		return t_point[:,0]

	def transformVec(A, mat):
		r =Voxeliser.transform(np.array([[A.x,A.y,A.z,0]]),mat)
		return vector.vector3D(r[0],r[1],r[2])

	def transformPoint(x,y,z,mat):
		point = Voxeliser.transform(np.array([[x,y,z,0]]),mat)
		return int(point[0]),int(point[1]),int(point[2])

	# Uses Transform the Coordinates
	def transformPoints(X,Y,Z,axis):

		mat = Model.model.getRotationMatrix(axis)
		for i in range(len(X)):
			t_point = Voxeliser.transform(np.array([[X[i],Y[i],Z[i],0]]),mat)# mat.dot(point.transpose())
			X[i]= t_point[0]
			Y[i]= t_point[1]
			Z[i]= t_point[2]

		return X,Y,Z

	#return List of points inside the triangle
	def getPointInsideTriangle(plane , pt1 , pt2 , pt3 , functionPLane ="XY"):

		ptA = Point.Point2D.getPoint2D(pt1,functionPLane)
		ptB = Point.Point2D.getPoint2D(pt2,functionPLane)
		ptC = Point.Point2D.getPoint2D(pt3,functionPLane)

		vertexes = [(ptA.x,ptA.y),(ptB.x,ptB.y),(ptC.x,ptC.y)]

		points=[] 

		#sort the list basis of x cord
		vertexes = sorted ( vertexes, key=operator.itemgetter(0))

		#get equation of Lines
		baseLine = Line.Line2D.getLine(vertexes[0],vertexes[2])

		for x in range (vertexes[0][0],vertexes[1][0],1):

			newLine = Line.Line2D.getLine(vertexes[0],vertexes[1])

			y_base = baseLine.getY(x)

			if y_base == math.inf:
				y_base = vertexes[0][1]

			y_new = newLine.getY(x)

			if y_new == math.inf:
				y_new = vertexes[1][1]

			for y in range(min(y_base,y_new),max(y_new,y_base)+1,1):
				
				# Determining the 3 coordinate
				z = plane.getZ(x,y)

				if functionPLane ==	"YZ":
					z= plane.getX(x,y)

				elif functionPLane == "XZ" :
					z = plane.getY(x,y)	

				points.append((x,y,z))

		for x in range ( vertexes[1][0],vertexes[2][0]+1,1):

			newLine = Line.Line2D.getLine(vertexes[1],vertexes[2])
			# newLine.printLine()

			# print(x)

			y_base = baseLine.getY(x)
			if y_base == math.inf:
				y_base = vertexes[1][1]

			y_new = newLine.getY(x)

			if y_new == math.inf:
				y_new = vertexes[2][1]

			# print (y_base,y_new)
			for y in range(min(y_base,y_new),max(y_new,y_base)+1,1):

				# Determining the 3 coordinate
				z = plane.getZ(x,y)

				if functionPLane ==	"YZ":
					z= plane.getX(x,y)

				elif functionPLane == "XZ" :
					z = plane.getY(x,y)	

				points.append((x,y,z))

		return points

	# returns disparityMap, colorImage, LightMap
	def getDisparitytMap( surface,l=1000,b=1000,h =1000, functionPLane ="XY", margin = float(1.3), colorImage = None,baseImage=None):

		## Variable for Light 
		h = int(h)
		cameraPos = vector.vector3D( b/2 , l/2 , h )
		cameraDir = vector.vector3D( 0 ,0 ,1 )

		lightSource = vector.vector3D(  b/2 , 1*l , h )
		LightDir = vector.vector3D.subVector(vector.vector3D(b/2,l/2,h/2),lightSource)
		LightPower = 50000
		# Intesity of Light to give effect of Shadders 
		LightMap = np.zeros((l,b), dtype = float)
		LightMap.fill(1)

		shadowMatrix = Model.model.getRotationMatrix(LightDir)
		shadowMap = np.zeros((1000,1000),dtype= float)

		# Displacement Map for size (x,y)
		disparityMap = np.zeros((l,b) ,dtype = float)

		if(type(colorImage) == type(None)):
			colorImage = np.zeros((l,b,3) ,dtype = np.uint8)
			colorImage.fill(255)


		X,Y,Z = Voxeliser.transformPoints(surface.X,surface.Y,surface.Z,cameraDir)

		Faces = surface.Faces
		Texture = surface.Texture

		minX = math.floor(min(X)) 
		minY = math.floor(min(Y))
		minZ = math.floor(min(Z))

		maxX = math.ceil(max(X))
		maxY = math.ceil(max(Y))
		maxZ = math.ceil(max(Z))

		constant = 0.000000000001
		minVector = vector.vector3D(minX,minY,minZ)

		scalex = ( b / (margin*(float(maxX-minX+constant))))
		scaley = ( l / (margin*float(maxY - minY+constant)))
		minScale = min( scalex,scaley)

		# computation of margin
		xMargin = (scalex/minScale)*margin
		xMargin = (xMargin -1)/(2*xMargin)

		yMargin = (scaley/minScale)*margin
		yMargin = (yMargin -1)/(2*yMargin)

		zMargin = (margin-1)/(2*margin)

		#Scale Factor
		ScaleVector = vector.vector3D( minScale, minScale , ( h/(margin*float(maxZ-minZ+constant))))
		# ScaleVector.printVector()
		TranslationVector = vector.vector3D(b*xMargin,l*yMargin,h*zMargin)
		i = 0
		for face in Faces:

			# colorValue = int((Texture[face[2]][0]+ Texture[face[2]][1]+Texture[face[2]][2])/3)
			colorValue = np.zeros(3)
			colorValue.fill(255)

			if(Texture != None):
				colorValue[0], colorValue[1] , colorValue[2] = Texture[face[0]][0],Texture[face[0]][1],Texture[face[0]][2]
			
			vec1 = vector.vector3D(X[face[0]],Y[face[0]],Z[face[0]])
			vec2 = vector.vector3D(X[face[1]],Y[face[1]],Z[face[1]])
			vec3 = vector.vector3D(X[face[2]],Y[face[2]],Z[face[2]])

			vec1 = vector.vector3D.addVector(vector.vector3D.vectorMultipy(ScaleVector,vector.vector3D.subVector(vec1,minVector)),TranslationVector)
			vec2 = vector.vector3D.addVector(vector.vector3D.vectorMultipy(ScaleVector,vector.vector3D.subVector(vec2,minVector)),TranslationVector)
			vec3 = vector.vector3D.addVector(vector.vector3D.vectorMultipy(ScaleVector,vector.vector3D.subVector(vec3,minVector)),TranslationVector)

			pt1 = vec1.vecToIntegerPoint()
			pt2 = vec2.vecToIntegerPoint()
			pt3 = vec3.vecToIntegerPoint()


			plane = Plane.Plane3D.getPlaneEquation( pt1, pt2 , pt3)

			## there are differenet type of Light,
			# Diffussed Light = 1/r^2 * Intensity * cosq
			# Ambient Light (from multiple reflection) = constant
			# specular light (where maximum reflection happens ) = cos(E ^ R) ^ 5 and Intensity
			# Follow http://www.opengl-tutorial.org/beginners-tutorials/tutorial-8-basic-shading/
			
			# to ease the computation will consider Centroid
			pos = Point.Point3D.getCentroid([pt1,pt2,pt3]).pointToVec()

			incidentRay = vector.vector3D.subVector(pos, lightSource)
			distance = incidentRay.getMagnitude()
			IntensityA = float (0.6)
			IntensityD = abs(1* ( LightPower * vector.vector3D.getCos(plane.normalVec,incidentRay))/(distance ** 2))
			reflectedRay = vector.vector3D.getReflectedVector(plane.normalVec, incidentRay)
			IntensityS = abs(1*( LightPower * (vector.vector3D.getCos(vector.vector3D(0,0,1),reflectedRay)** 5) )/(distance ** 2) )
			Intensity =IntensityS + IntensityD + IntensityA
			Intensity = 1 if Intensity > 1 else Intensity
			# if type(baseImage) != None:
			# 	Intensity = 1

			points = Voxeliser.getPointInsideTriangle(plane, pt1 , pt2 , pt3 )
			# print(i)

			for (x,y,z) in points:

				# Computation for Shader
				s_x,s_y,s_z = Voxeliser.transformPoint(x,y,z,shadowMatrix)
				value = int(z)
				# print(s_x,s_y)
				s_y = l-1-s_y
				
				if s_y >= 0 and s_y < 1000 and s_x >=0 and s_y < 1000:
					# print("hollah")
					shadowMap[s_y][s_x] = max(value,shadowMap[s_y][s_x])

				# Computation on  Intensity Map
				value = int ( z )
				if value > disparityMap[l-1-y][x]:
					disparityMap[l-1-y][x]= value
					colorImage[l-1-y][x] = [np.uint8(abs(colorValue[0]*Intensity)),np.uint8(abs(colorValue[1]*Intensity)),np.uint8(abs(colorValue[2]*Intensity))]
					LightMap [l-1-y][x] = Intensity

		
		return disparityMap, colorImage, LightMap ,shadowMap


















