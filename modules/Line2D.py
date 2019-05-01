import Point2D

class Line2D:

	# general Equation of line ax + by +c =0
	def __init__ (self,A,B):
		self.pt1 = A
		self.pt2 = B

		self.a , self.b , self.c  = self.getLine()

	def getLine(self):

		x1 = self.pt1.x
		y1 = self.pt1.y

		x2 = self.pt2.x
		y2 = self.pt2.y

		a = (y2-y1)
		b = (x1-x2)
		c = -1*(x1*(a) + y1*(b))

		return a,b,c

	def getY(self, x):

		if self.b == 0:
			if self.a*x == int (-1*self.c):
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
