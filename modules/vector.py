import math
from modules import Point

class vector2D:

	def __init__(self, x = 0, y =0 ):
		self.x = (x)
		self.y = (y)

	def unitVector(self):

		mag = self.getModulus()
		if mag == float(0):
			return (vector2D())

		return vector2D(self.x/mag, self.y/mag)


	def getModulus(self):
		return math.sqrt( self.x **2 + self.y **2)

	## return vector2D
	def addVector( self, A):
		return ( vector2D(self.x+A.x,self.y+A.y))

	def scaledVector(self ,mag):
		n = self.unitVector()
		return (vector2D( mag * n.x , mag * n.y ))

	def printVector(self):
		print ( self.x , self.y)


class vector3D:

	def __init__(self, x=0, y=0, z=0):
		self.x = x
		self.y = y
		self.z = z

	def getVector3D( pt2, pt1):

		vec = vector3D()
		vec.x = pt1.x - pt2.x
		vec.y = pt1.y - pt2.y
		vec.z = pt1.z - pt2.z
		return (vec)

	def getMagnitude ( self):
		return math.sqrt( self.x **2 + self.y **2 + self.z **2)

	def getUnitVector(A):

		mag = A.getMagnitude()
		if mag == float(0):
			return (vector3D())

		return vector3D(A.x/mag, A.y/mag , A.z/mag)

	def crossProduct( vectorA, vectorB):

		a = vectorA.y*vectorB.z - vectorA.z * vectorB.y
		b =	vectorA.z*vectorB.x - vectorA.x * vectorB.z
		c = vectorA.x*vectorB.y - vectorA.y * vectorB.x

		return vector3D(a,b,c)

	def dotProduct(self,vec):
		return (self.x*vec.x + self.y*vec.y + self.z*vec.z)

	def addVector(A,B):
		return vector3D( A.x+B.x , A.y+B.y , A.z+B.z )

	def subVector(A,B):
		return vector3D( A.x-B.x , A.y-B.y , A.z-B.z )

	def scalerMultpy(A, mag):
		return vector3D( mag * A.x, mag * A.y, mag * A.z)

	def vectorMultipy(A, B):
		return vector3D( A.x*B.x , A.y*B.y , A.z*B.z)

	def vecToIntegerPoint(self):
		return Point.Point3D(int(round(self.x,0)) , int(round(self.y,0)) , int(round(self.z,0)))

	def vecToPoints(self, decimal):
		return Point.Point3D(round(self.x,decimal) , round(self.x,decimal) , round(self.x,decimal) )

	def getVector(self):
		return vector3D(self.x,self.y,self.z)

	def printVector(self):
		print ( self.x , self.y, self.z)	


	def getCos(A, B):

		A = vector3D.getUnitVector(A)
		B = vector3D.getUnitVector(B)

		return A.dotProduct(B)

	def getReflectedVector(normal , incident):

		normal = vector3D.getUnitVector(normal)
		cos = vector3D.getCos( incident, normal)

		magnitute = incident.getMagnitude()

		vecReflectedAlongNormal = vector3D.scalerMultpy(normal, float(-2)*cos * magnitute )
		
		return vector3D.addVector(incident, vecReflectedAlongNormal)
































