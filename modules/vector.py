import math

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

	def getVector3D(self,pt2, pt1):

		self.x = pt1[0] - pt2[0]
		self.y = pt1[1] - pt2[1]
		self.z = pt1[2] - pt2[2]

		return (getVector())

	def getVector(self):
		return vector3D(self.x,self.y,self.z)

	def printVector(self):
		print ( self.x , self.y,self.z)	