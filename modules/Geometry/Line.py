from modules.Geometry import Point
from modules.Utils import vector
import math

class Line2D:

	# general Equation of line ax + by +c =0
	def __init__ (self):

		self.a = 0
		self.b = 0
		self.c = 0

	def newLine ( a, b, c):
		ln = Line2D()
		ln.a = a
		ln.b = b
		ln.c = c
		return ln 

	def getLine( pt1 , pt2):

		line = Line2D()

		x1 = pt1[0]
		y1 = pt1[1]

		x2 = pt2[0]
		y2 = pt2[1]

		line.a = (y2-y1)
		line.b = (x1-x2)
		line.c = -1*(x1*(line.a) + y1*(line.b))

		return line

	def getLinePoint(pt1, pt2):
		return Line2D.getLine((pt1.x,pt1.y),(pt2.x,pt2.y))

	def getY(self, x):

		if self.b == 0:
			if self.a * x == int (-1*self.c):
				return  math.inf
			else :
				return None

		return int(-1*round((float(self.a*x+self.c)/float(self.b)),0))


	def getX (self,y):

		if self.a == 0:
			if self.b*y == int (-1*self.c):
				return  math.inf
			else :
				return None

		return int(-1*round((float(self.b*y+self.c)/float(self.a)),0))

	def getDistance(self, point):

		mod = math.sqrt( self.a**2  + self.b**2)
		value = float (abs( self.a*point.x + self.b*point.y + self.c ))
		return value/mod;

	def equate ( eq1, eq2):
		base = eq2.b * eq1.a - eq1.b * eq2.a
		num1 = eq1.c * eq2.a - eq2.c * eq1.a # num of y
		num2 = eq1.c * eq2.b - eq2.c * eq1.b # num of x

		#case many solution 
		if ( base == 0 and num1 ==0 ):
			return 1, -1*((eq1.c + eq1.a)/eq1.b)
		if ( base == 0 and num1 !=0 and num2 !=0):
			return None , None
		if ( base != 0):
			return num2/base, -1*num1/base 


	def printLine(self):
		print (self.a, self.b, self.c)


class Line3D:

	def __init__ (self):
		self.pos = vector.vector3D()
		self.dir = vector.vector3D()

	def line3D(pt1, pt2):
		self.pos = pt1.pointToVec()
		self.dir = vector.vector3D.getUnitVector(vector.vector3D.getVector3D(pt1, pt2))

	def line3D( vec1 , vec2 ):
		ln = Line3D()
		ln.pos = vec1
		ln.dir = vector.vector3D.getUnitVector(vector.vector3D.subVector(vec1, vec2))
		return ln

	def newLine ( pos, direction ):
		ln = Line3D()
		ln.pos = pos
		ln.dir = vector.vector3D.getUnitVector(direction )
		return ln 

	def onLine ( vec1 , line ):
		temp = vector.vector3D.subVector(line.pos , vec1)
		return vector.vector3D(temp, line.dir)

	def isParallel ( ln1 , ln2):
		return vector.vector3D.isParallel(ln1.dir,ln2.dir)

	def isConcurrent ( ln1 , ln2  ):
		if (isParallel(ln1, ln2) and onLine(ln1, ln2.pos)):
			return True
		return False

	def getDistancePoint ( ln , pt ):
		return getDistance(ln, pt.pointToVec())

	def getDistance ( ln , vec):
		vec = vector.vector3D.subVector(vec, ln.pos)
		return vector.vector3D.crossProduct(vec, ln.dir).getMagnitude()

	def getDistanceLines ( ln1, ln2 ):
		vec = vector.vector3D.crossProduct(ln1.dir, ln2.dir)
		return vector.vector3D.crossProduct(vec, vector.vector3D.subVector(ln1.pos,ln2.pos)).getMagnitude()

	def printLine(self):
		print("Axis")
		self.dir.printVector()
		print("Position")
		self.pos.printVector()

	def getMinimumDisPoints (ln1, ln2):

		a = ln1.dir.dotProduct(ln2.dir) 

		c1 = ln1.dir.dotProduct(ln1.pos) - ln1.dir.dotProduct(ln2.pos)
		c2 = ln2.dir.dotProduct(ln1.pos) - ln2.dir.dotProduct(ln2.pos)
		"""
			ln1.dir += (di,0,dj)
						delta c1


		"""

		"""
		tTo solve the equation we consider two arbitary point on the system 
		 lamda nad nue two variable
		(a1 - a2).L1 =0 
		(a1 -a2).L2 = 0
		a1.L1 = L1.L1 x - pos1.L1 = 0
		a2.L1 = L2.L1 x - pos2.L1 = 0

		therefore 
		eq1 =  x - ay = c1 
		eq2 = ax -  y = c2
		solution of x and y are value of lambda and nue
		"""

		eq1 = Line2D.newLine(1, -1*a , -1*c1)
		eq2 = Line2D.newLine(a, -1, -1*c2 )
		t , u = Line2D.equate( eq1 ,eq2)

	    # if no nearest point exist 
		if ( t == None and u == None):
			return None , None


		vec1 = vector.vector3D.addVector(ln1.pos,vector.vector3D.scalerMultpy(ln1.dir, t))
		vec2 = vector.vector3D.addVector(ln2.pos, vector.vector3D.scalerMultpy(ln2.dir, u))

		return vec1 , vec2


