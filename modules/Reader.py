import os.path
from pathlib import Path
from PIL import Image
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
			face.append(int(A[i])-1)
			face.append(int(B[i])-1)
			face.append(int(C[i])-1)
			Faces.append(face)

		return Faces

	def readTexture(self,fileName):

		A,B,C =self.readVertexes(fileName)

		texture =[]
		for i in range(len(A)):
			colors=[]
			colors.append(int(A[i]))
			colors.append(int(B[i]))
			colors.append(int(C[i]))
			texture.append(colors)

		return texture


class writer:
	"""docstring for write"""
	def __init__(self):

		self.currentPath=os.getcwd()
		os.chdir('results')
		pathNew=Path(os.getcwd())
		os.chdir(self.currentPath)
		self.pathResult = pathNew
		
	def writeImage (self,fileName,result):

		filePath = self.pathResult / fileName
		img = Image.fromarray(result)
		img.save (filePath)







