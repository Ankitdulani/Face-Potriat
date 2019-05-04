import numpy as np
import math
from modules import Point
from modules import vector
from modules import Plane
from modules import Line
from modules import Surface
import operator

class Voxeliser:

	def __init__ ( self ):
		print ( "Voxeliser Created" )

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

	def getDisparitytMap( surface,l=900,b=1000,h =1000, functionPLane ="XY"):

		X = surface.X
		Y = surface.Y
		Z = surface.Z

		Faces = surface.Faces

		Texture = surface.Texture

		# Displacement Map for size (x,y)
		disparityMap = np.zeros((l,b) ,dtype = float)

		colorImage = np.zeros((l,b,3) ,dtype = np.uint8)
		colorImage.fill(255)

		minVector = vector.vector3D(float(-1),float(-1),float(-1))
		maxZ = max(Z)

		#Scale Factor
		ScaleVector = vector.vector3D(int ( b / float(2)),int ( l / float(2)),int ( h/(maxZ - float(-1))))

		i = 0

		for face in Faces:

			# colorValue = int((Texture[face[2]][0]+ Texture[face[2]][1]+Texture[face[2]][2])/3)
			colorValue = [Texture[face[2]][0],Texture[face[2]][1],Texture[face[2]][2]]
			# print(colorValue)
			
			vec1 = vector.vector3D(X[face[0]],Y[face[0]],Z[face[0]])
			vec2 = vector.vector3D(X[face[1]],Y[face[1]],Z[face[1]])
			vec3 = vector.vector3D(X[face[2]],Y[face[2]],Z[face[2]])

			vec1 = vector.vector3D.vectorMultipy(ScaleVector,vector.vector3D.subVector(vec1,minVector))
			vec2 = vector.vector3D.vectorMultipy(ScaleVector,vector.vector3D.subVector(vec2,minVector))
			vec3 = vector.vector3D.vectorMultipy(ScaleVector,vector.vector3D.subVector(vec3,minVector))

			pt1 = vec1.vecToIntegerPoint()
			pt2 = vec2.vecToIntegerPoint()
			pt3 = vec3.vecToIntegerPoint()

			plane = Plane.Plane3D.getPlaneEquation( pt1, pt2 , pt3)

			points = Voxeliser.getPointInsideTriangle(plane, pt1 , pt2 , pt3 )

			# print (len(points))

			for (x,y,z) in points:
				value = int ( z )
				if value > disparityMap[l-1-y][x]:
					disparityMap[l-1-y][x]= value
					colorImage[l-1-y][x] = colorValue

		return disparityMap, colorImage


















