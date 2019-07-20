import math
from modules import vector
import numpy as np

class surface:

	def __init__ (self, X,Y,Z, Faces, texture = None):
		self.X = X
		self.Y = Y
		self.Z = Z

		self.Faces = Faces
		self.Texture = texture
		self.Edges = []

	def convertTextureForFaces(self):

		texture = self.Texture
		index = 0
		newTexture = []
		
		for face in self.Faces:

			colorR = int ((texture[face[0]][0])) #+ texture[face[1]][0] + texture[face[2]][0])/3)
			colorB = int ((texture[face[0]][1])) #+ texture[face[1]][1] + texture[face[2]][1])/3)
			colorG = int ((texture[face[0]][2])) #+ texture[face[1]][2] + texture[face[2]][2])/3)

			# print((colorR,colorG,colorB))
			newTexture.append((colorR,colorG,colorB))

		self.Texture = newTexture

	# triangulagted faces are no overlapping
	# Edges are also not Partially Shared
	# Return the List of Tuple with indexes of Points
	def setEdges(self):

		self.Edges=[]

		for i in range(len(self.Faces)):
			self.Edges.append((min(self.Faces[i][0],self.Faces[i][1]),max(self.Faces[i][0],self.Faces[i][1])))
			self.Edges.append((min(self.Faces[i][1],self.Faces[i][2]),max(self.Faces[i][1],self.Faces[i][2])))
			self.Edges.append((min(self.Faces[i][0],self.Faces[i][2]),max(self.Faces[i][0],self.Faces[i][2])))

	def getEdges(self):

		if len(self.Edges) == 0:
			self.setEdges()

		return self.Edges

	def getBoundaryEdges(self):

		self.setEdges()
		print(len(self.Edges))

		hashMapParllelEdges=dict()
		boundaryEdges=[]

		for Edge in self.Edges:

			# creating key for hash map 
			l=abs(self.X[Edge[0]]-self.X[Edge[1]])
			m=abs(self.Y[Edge[0]]-self.Y[Edge[1]])
			n=abs(self.Z[Edge[0]]-self.Z[Edge[1]])

			mod = math.sqrt(l** 2 + m**2 +n**2)

			l = str("{0:.6f}".format(l/mod))
			m = str("{0:.6f}".format(m/mod))
			n = str("{0:.6f}".format(n/mod))

			key =l+m+n

			#Checking for parllel edges
			#Also removing the edge if used twice in the code.
			if hashMapParllelEdges.get(key) == None:
				hashMapParllelEdges[key] = [Edge]
			else :
				if (Edge) in hashMapParllelEdges[key]:
					hashMapParllelEdges[key].remove((Edge))
				else:	
					hashMapParllelEdges[key].append(Edge)

				if hashMapParllelEdges[key] == []:
					del hashMapParllelEdges[key]

		for key, value in hashMapParllelEdges.items():
			boundaryEdges.append(value[0])


		return 	boundaryEdges

	## This Fucntion return extended Surface which is connected to Base Plane
	def getExtendedSurfaceToBase(self,boundaryEdges):

		newSurface = surface(self.X, self.Y,self.Z,self.Faces)

		indexV= len(newSurface.X)
		# Zmin = float (-1.000000000000000000)
		Zmin = float(min(newSurface.Z) -0.000000000000000001)

		for (pt1, pt2) in boundaryEdges:

			## added (X,Y,Z) in the list at the end
			(newSurface.X).append(newSurface.X[pt1])
			(newSurface.Y).append(newSurface.Y[pt1])
			(newSurface.Z).append(Zmin)

			(newSurface.X).append(newSurface.X[pt2])
			(newSurface.Y).append(newSurface.Y[pt2])
			(newSurface.Z).append(Zmin)

			## added two triangle which 

			(newSurface.Faces).append([pt1, indexV, indexV+1])
			(newSurface.Faces).append([pt1,pt2,indexV+1])

			indexV+=2



