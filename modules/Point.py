from modules import vector

class Point2D:

	def __init__ (self, x =0, y=0 ):

		self.x = x
		self.y = y

	def getPoint2D(pt, fucntionalPlane="XY"):

		point = Point2D(pt.x,pt.y)

		if fucntionalPlane == "YZ":
			point = Point2D( pt.y, pt.z)

		elif fucntionalPlane == "XZ":
			point = Point2D( pt.x, pt.z)

		return point

	def printPoint(self):
		print (self.x , self.y)

class Point3D:

	def __init__ (self, x =0, y=0 , z=0):

		self.x = x
		self.y = y
		self.z = z

	def pointToVec(self):
		return vector.vector3D(self.x , self.y , self.z) 

	def addPoint(self, pt):
		return Point3D(self.x+pt.x, self.y+pt.y , self.z+pt.z)

	def print(self):

		(self.pointToVec()).printVector()

	def getCentroid(Points):

		pt = Point3D()
		for point in Points:
			pt = pt.addPoint(point)

		n = len (Points)

		return Point3D( int(pt.x/n) , int (pt.y/n) , int (pt.z/n))
