from modules.Utils import vector
from modules.Geometry import Line
from modules.Utils import BresenhamAlgo
import math
import cv2


class Cylinder:


	def __init__(self):
		self.R =1
		self.pos = vector.vector3D(0,0,1)
		# -1 defines infinity
		self.H = -1
		self.Axis = vector.vector3D(0,0,1)

	def setRadius( self, r ):
		self.R = r

	def setAxis (self, axis ):
		self.Axis = axis

	def setPosition(self, pos):
		self.pos = pos

	def newCylinder(r, pos,axis):
		c = Cylinder()
		c.setRadius(r)
		c.setPosition(pos)
		c.setAxis(axis)
		return c

	def getIntersectionWithLine( self , ln ):

		axisLine = Line.Line3D.newLine(self.pos, self.Axis)
		# print("\nLine Values")
		# ln.printLine()
		# axisLine.printLine()
		# print("\n")


		# new incident += (di,0,dj)
		vecA, vecB = Line.Line3D.getMinimumDisPoints(ln, axisLine)
		#vecA.printVector()
		# now we have two point 
		minDis = vector.vector3D.subVector(vecA, vecB).getMagnitude()

		#print(minDis)
		
		if self.R**2 - minDis**2 < 0 :
			return None 
		#using pythogoras at cylindercial Interface
		a = math.sqrt( self.R**2 - minDis**2 )

		vec = vector.vector3D.subVector(ln.pos, vecA)
		b = vector.vector3D.getPerpendicularComponent(vec, self.Axis).getMagnitude()

		intersectionPoint = vector.vector3D.addVector( vecA , vector.vector3D.scalerMultpy(vec, float(a)/float(b)))

		return intersectionPoint

	def getNoramlVec ( self, pt,img):

		# pt.printVector()

		# img = cv2.circle(img,(int(pt.x) , int(pt.y)),20,(0,0,255),3)

		# img = BresenhamAlgo.bresenhamLine.drawLine(img,0,
		# 	[(int(pt.x),int(pt.y)),(int(self.pos.x),int(self.pos.y))])
		vec = vector.vector3D.getPerpendicularComponent(
								vector.vector3D.subVector(pt,self.pos),
								self.Axis)

		# vec.printVector()
		return vec,img





