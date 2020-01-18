import os.path
import cv2
from pathlib import Path
from PIL import Image
import numpy as np
import skimage.io
class reader:

	def __init__(self):

		self.currentPath=Path(os.getcwd())
		os.chdir('resources')
		path=Path(os.getcwd())
		os.chdir(self.currentPath)
		self.pathResource = path

		self.v_prefix = "vertex"
		self.v_suffix = "_new.txt"
		self.f_prefix = "face"
		self.f_suffix = "_new.txt"
		self.t_prefix = "texture"
		self.t_suffix = "_new.txt"

		self.inter_result = "temp"

		print ("** Reader Intiatlised **")

	# return the X,Y,Z value for this is conceptuale
	def readFile(self,name,fileName):

		path = self.pathResource / name / fileName
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

	def readVertexes(self, name):
		fileName = str(self.v_prefix)+name+str(self.v_suffix)
		return self.readFile(name,fileName)

	def readFaces(self,name):

		fileName = str(self.f_prefix)+name+str(self.f_suffix)
		A,B,C = self.readFile(name,fileName)

		Faces=[]
		for i in range(len(A)):
			face=[]
			face.append(int(A[i])-1)
			face.append(int(B[i])-1)
			face.append(int(C[i])-1)
			Faces.append(face)

		return Faces

	def readTexture(self,name):

		fileName = str(self.t_prefix)+name+str(self.t_suffix)
		A,B,C = self.readFile(name,fileName)

		texture =[]
		for i in range(len(A)):
			colors=[]
			colors.append(int(A[i]))
			colors.append(int(B[i]))
			colors.append(int(C[i]))
			texture.append(colors)

		return texture

	def parseOBJ(self, fileName):

		fileName +=".obj"
		path = self.pathResource / "obj" / fileName
		file = open(path, 'r')

		vertexesX = []
		vertexesY = []
		vertexesZ = []
		Faces = []
		Texture = []

		
		print("reading the files")

		for line in file:
			x = self.noEmptySplit(line.strip())
			# print(x)

			if(len(x) == 0):
				continue

			if x[0] == "v":
				vertexesX.append(float(x[1]))
				vertexesY.append(float(x[2]))
				vertexesZ.append(float(x[3]))
				if len(x) > 6:
					Texture.append([round(255*float(x[4])),round(255*float(x[5])),round(255*float(x[6]))])

			elif x[0] == "f":
				v = []
				for i in range(1,len(x)):
					v.append(int(str(x[i]).split('/')[0]) -1)

				for i in range(1,len(v)-1,1):
					Faces.append([v[0],v[i],v[i+1]])
				
		Texture = None if len(Texture) == 0 else Texture
		
		return vertexesX,vertexesY,vertexesZ,Faces,Texture

	def readImage(self,name,img_prefix = "img_",img_suffix = ".jpg",folder="temp"):
		
		base_path = self.pathResource / "images"
		if folder == "Text":
			base_path = self.pathResource / "Text"

		
		if folder == "anomorphosis":
			base_path = self.pathResource / "anomorphosis"
		
		fileName = img_prefix+str(name)+img_suffix
		filePath = base_path / fileName

		if folder == "temp":
			filePath = self.currentPath / self.inter_result / fileName
			
		# return Image.open(str(filePath))
		#print(str(filePath))
		img = cv2.imread(str(filePath))
		
		# print(img.shape)
		# return img
		return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		# return  skimage.io.imread(str(filePath))

	def noEmptySplit(self,x):
		x = x.split(' ')
		result = []
		for i in x:
			# print(i)
			if i!='':
				result.append(i)
		return result

	def readNpArray(self,fileName):
		filePath = self.currentPath / self.inter_result / fileName
		return np.loadtxt(filePath, delimiter=",")

	def checkIfExist(self,fileName):
		filePath = Path(self.currentPath) / self.inter_result /fileName
		return os.path.exists(filePath)


class writer:
	"""docstring for write"""
	def __init__(self):

		self.currentPath=Path(os.getcwd())
		os.chdir('results')
		pathNew=Path(os.getcwd())
		os.chdir(self.currentPath)
		self.pathResult = pathNew

		self.inter_result = "temp"

	def checkIfExits(self,folder):
		path = self.pathResult / folder
		return os.path.exists(path)
		
	def writeImage (self,fileName,result,folder=""):

		filePath = Path()
		if folder == "temp":
			filePath = self.currentPath / self.inter_result / fileName

		else:
			if self.checkIfExits(folder) == False:
				os.mkdir(self.pathResult / folder)
			filePath = self.pathResult / folder /fileName

		# img = Image.fromarray(result)
		# print("writinhg",type(result))
		# print(fileName , result.shape)
		if len(result.shape) > 2:
			result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
		cv2.imwrite(str(filePath),result)
		# 
		# img.save (filePath)

	def writeNpArray(self,fileName,result):

		filePath = self.currentPath / self.inter_result / fileName
		np.savetxt(filePath, result, fmt="%s", delimiter=",")







