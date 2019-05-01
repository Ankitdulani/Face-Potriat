from modules import Point
import math

class Line2D:

	# general Equation of line ax + by +c =0
	def __init__ (self):

		self.a = 0
		self.b = 0
		self.c = 0

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



	def printLine(self):
		print (self.a, self.b, self.c)