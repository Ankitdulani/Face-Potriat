import os.path
from pathlib import Path
class reader:

	def __init__(self):

		self.currentPath=os.getcwd()
		os.chdir('resources')
		path=Path(os.getcwd())
		os.chdir(self.currentPath)
		self.pathResource = path

		print ("** Reader Intiatlised **")

	def readVertexes(self, fileName):

		path = self.pathResource / fileName
		file =open(path,'r')

		A=[]
		B=[]
		C=[]

		for line in file:
			x=line.strip().split(',')
			A.append(float(x[0]))
			B.append(float(x[1]))
			C.append(float(x[2]))

		return A,B,C

	def readFaces(self,fileName):

		A,B,C =self.readVertexes(fileName)

		Faces=[]
		for i in range(len(A)):
			face=[]
			face.append(int(A[i]))
			face.append(int(B[i]))
			face.append(int(C[i]))
			Faces.append(face)

		return Faces