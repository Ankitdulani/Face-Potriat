class vector3D:

	def __init__(self, x=0, y=0, z=0):
		self.x = x
		self.y = y
		self.z = z

	def getVector3D(self,pt1, pt2):

		self.x = pt1.x - pt2.x
		self.y = pt1.y - pt2.y
		self.z = pt1.z - pt2.z

		return (getVector())

	def getVector(self):
		return (self.x,self.y,self.z)