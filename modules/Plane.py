from modules import vector

class Plane3D:

	# General Equation of Vector ax +by+cz +d =0
	def __init__ (self):
		self.normalVec = vector.vector3D()
		self.d = 0

	def getPlaneEquation(pt1,pt2,pt3):

		plane = Plane3D()

		vectorA = vector.vector3D.getVector3D(pt3,pt1)
		VectorB = vector.vector3D.getVector3D(pt2,pt1)

		plane.normalVec = vector.vector3D.crossProduct(vectorA,VectorB)

		posVector = pt1.pointToVec()

		plane.d = -1*(plane.normalVec).dotProduct(posVector)

		return (plane)

	def printEquation(self):

		print ("normal Vector" )
		self.normalVec.printVector()
		print ("constant" , self.d )
		

	def getZ(self,x,y):

		vec = vector.vector3D(x,y,0)
		val = self.normalVec.dotProduct(vec) + self.d

		if (self.normalVec.z) == float(0):
			return float(0)

		return float(-1* (val/self.normalVec.z))

	def getX(self,y,z):

		vec = vector.vector3D(0,y,z)
		val = self.normalVec.dotProduct(vec) + self.d

		if (self.normalVec.x) == float(0):
			return float(0)

		return float(-1* (val/self.normalVec.x))

	def getY(self,x,z):

		vec = vector.vector3D(x,0,z)
		val = self.normalVec.dotProduct(vec) + self.d

		if (self.normalVec.y) == float(0):
			return float(0)

		return float(-1* (val/self.normalVec.y))