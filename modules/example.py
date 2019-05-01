import plotly
import plotly.offline as py
import plotly.figure_factory as FF
import plotly.graph_objs as go
import math 
import random
import numpy as np
from PIL import Image
from operator import itemgetter
from scipy.spatial import Delaunay

import vector


#Return a tuple
def readVertexs(pathName):

	file =open(pathName,'r')

	A=[]
	B=[]
	C=[]

	for line in file:
		x=line.strip().split(',')
		A.append(float(x[0]))
		B.append(float(x[1]))
		C.append(float(x[2]))

	return A,B,C

# Returns a tuple
def readFaces(pathName):

	A,B,C =readVertexs(pathName)

	Faces=[]
	for i in range(len(A)):
		face=[]
		face.append(int(A[i]))
		face.append(int(B[i]))
		face.append(int(C[i]))
		Faces.append(face)

	return Faces

#return the list of edges
def getEdges(Faces):
	Edges=[]

	for i in range(len(Faces)):
		Edges.append((min(Faces[i][0],Faces[i][1]),max(Faces[i][0],Faces[i][1])))
		Edges.append((min(Faces[i][1],Faces[i][2]),max(Faces[i][1],Faces[i][2])))
		Edges.append((min(Faces[i][0],Faces[i][2]),max(Faces[i][0],Faces[i][2])))

	return Edges

#return Dict Edges with same Slopes
# dictonary holds the key that is slope and List of Edges
def getBoundaryEdges(X,Y,Z, Edges , perc=6):

	hashMapParllelEdges=dict()

	boundaryEdges=[]

	for Edge in Edges:

		# creating key for hash map 
		l=abs(X[Edge[0]]-X[Edge[1]])
		m=abs(Y[Edge[0]]-Y[Edge[1]])
		n=abs(Z[Edge[0]]-Z[Edge[1]])

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

## Function adds Faces and vertexes into the list 
## These addition Faces connect surface to the Boundary
def getClosedSurface(Faces,X,Y,Z, boundaryEdges):

	indexV= len(X)
	# Zmin = float (-1.000000000000000000)
	Zmin = float(min(Z) -0.000000000000000001)

	for (pt1, pt2) in boundaryEdges:

		## added (X,Y,Z) in the list at the end
		X.append(X[pt1])
		Y.append(Y[pt1])
		Z.append(Zmin)

		X.append(X[pt2])
		Y.append(Y[pt2])
		Z.append(Zmin)

		## added two triangle which 

		Faces.append([pt1, indexV, indexV+1])
		Faces.append([pt1,pt2,indexV+1])

		indexV+=2

	return Faces,X,Y,Z

#This function extract l, m ,n  from the string
def getSlope(slope):

	s= slope.split('.')
	l=float(s[1])/10000000.0
	m=float(s[2])/10000000.0
	n=float(s[3])/1000000.0


	if l==0 :
		l=0.000001

	if m ==0:
		m=0.000001

	if n==0:
		n=0.000001

	return l,m,n

def checkIfSkew(A,B,slope):
	isSkew=True
	alpha = []
	for i in range(3):
		alpha.append((A[i]-B[i])/slope[i])

	if (alpha[0]==alpha[1] and alpha[0]==alpha[2]):
		isSkew=False

	return bool(isSkew)

#this function segregate the flase parllel(Skew Lines) and parllel lines
def getSkewLines(X,Y,Z, parllelLinesList):

	groupList=dict()
	count=1

	print(len(parllelLinesList))

	farzi=None
	maxP=1
	
	for slope,value in  parllelLinesList.items():

		parlleLines=[]
		parllelLines=value


		l,m,n= getSlope(slope)

		# print (farzi)
		if maxP <= len(value):
			maxP=len(value)
			farzi=slope



		for edge in parllelLines:

			# print (type(parllelLines[i]))
			# edge=parlleLines[i]
			# print(edge)
			break

			if bool(groupList) == False:
				groupList[count]=[edge]
				count+=1
				continue

			isSkew =True

			# print("in Parllel Lined loop")
			#if edge is parllel to any of the element in group list 
			for key, lines in groupList.items():

				# print("creating an edge list")
				Lines=[]
				Lines=lines
				# print(type(Lines))
				baseEdge=Lines[0]
				# print (type(baseEdge))

				if checkIfSkew((X[baseEdge[0]],Y[baseEdge[0]],Z[baseEdge[0]]), (X[edge[0]],Y[edge[0]],Z[edge[0]]),(l,m,n)) == False:
					groupList[key].append(edge)
					isSkew = False
					break

			if isSkew == True:
				groupList[count]=[edge]
				count+=1

	print(maxP)
	print(parllelLinesList[farzi])
	return groupList

# Return dictionary of edges with same orienatation 
def getIntervals( X,Y,Z, faces ):

	intervals=dict()

	maxEle=1

	for i in range(len(X)):

		face=Faces[i]
		A=[0,1,0]
		B=[1,2,2]

		for j in range(3):
			key= getKey(X,Y,Z ,face[A[j]],face[B[j]])

			if key == "":
				continue

			interval =((i,int(A[j]), int (B[j])))

			if  intervals.get(key) == None:
				intervals[key] = [interval]
				
			else:
				intervals[key].append(interval)
				maxEle=max(maxEle,len(intervals[key]))
				# print("sucked")

	print (len(intervals),len(faces))
	print ("maximum number of element in intervals", maxEle)
	return intervals

# Return the list of edge which are concurrent
def getConcurrentEdges(interval,slope, faces, X, Y, Z):

	s= slope.split('.')
	l=float(s[1])/100000.0
	m=float(s[2])/100000.0
	n=float(s[3])/10000.0

	if l==0 :
		l=0.0001

	if m ==0:
		m=0.0001

	if n==0:
		n=0.0001
	
	parllelEdges=[]
	parllelLines=0

	for i,a,b in interval:

		if len(parllelEdges)==0:
			parllelEdges.append([(X[faces[i][a]],Y[faces[i][a]],Z[faces[i][a]],i,a,b)])
			parllelLines +=1

		else:
			x=X[faces[i][a]]
			y=Y[faces[i][a]]
			z=Z[faces[i][a]]

			countA=0
			countB=0

			for o in range(len(parllelEdges)):
				
				x1,y1,z1= parllelEdges[o][0][0:3]
				e=float(x-x1)/l
				f=float(y-y1)/m
				g=float(z-z1)/n


				if e!=0 and f!=0 and g!=0:
					countA+=1
				else:
					countB+=1
				# print (e,f,g)
				# if e==f and e==g:
				# 	parllelEdges[o].append((x,y,z,i,a,b))
				# else:
				# 	parllelEdges.append((x,y,z,i,a,b))

			print ("countA", countA)
	return parllelEdges

#Return a Dictionary of boundary Points
def getBoundaryPoints( X,Y,Z , faces):


	intervals = getIntervals(X,Y,Z ,faces)

	for slope,interval  in intervals.items():
		x= getConcurrentEdges(interval, slope, faces, X,Y,Z)




	## Expore the hash map and find the intervel wihch is partial occupied

# Return Random list of y corrdinated for lines
def getRandomLinesList( len,start=1,maxInterval=5):

	randomLines=[]
	x = start
	while x < len + start:
		randomLines.append(x)
		x+= random.randrange(0,maxInterval,1)

	return randomLines

#Return tuplte of faces
def getFilteredFaces(Faces, boundaryPoints):


	FilteredFaces=[] 

	for vertices in Faces:
		count = 0

		if vertices[0] in boundaryPoints:
			count +=1
		if vertices[1] in boundaryPoints:
			count +=1
		if vertices[2] in boundaryPoints:
			count +=1

		if count < 1:	
			FilteredFaces.append(vertices)

	return FilteredFaces

#Return 2D Array 
def createTestDisplacementMap( key, l=900,b=900):	

	displacementMap = np.zeros((l,b) ,dtype = int)
	maxHeight = 300

	## Creates A cube
	if key == "cube":
		displacementMap[300:600,300:600]=int(maxHeight)

	## Creating Displacement Map for Wedge
	if key =="wedge":
		for i in range (300, 600,1):
			displacementMap[i,300:600] = int (maxHeight-(i-int(450)))

	if key == "hemisphere":
		centerY = 450
		centerX = 450
		centerZ = 0
		radius = 150

		# General Equation for sphere is x^2 + y^2 + z^2 =r^2
		for i in range(0, radius +1,1):
			for j in range(0, radius+1,1):

				value = radius**2 - i**2 -j**2

				if value <0:
					continue
				alpha = int (round (math.sqrt(value),0))
				displacementMap[i+centerY][j+centerX]= alpha
				displacementMap[centerY-i][centerX-j]= alpha
				displacementMap[centerY+i][centerX-j]= alpha
				displacementMap[centerY-i][centerX+j]= alpha

	return displacementMap


### Use Line2D.py
def getLine(A,B):

	# general Equation of line ax + by +c =0
	x1=A[0]
	y1= A[1]

	x2 = B[0]
	y2 = B[1]

	a = (y2-y1)
	b = (x1-x2)
	c = -1*(x1*(a) + y1*(b))

	return (a,b,c)

#### Use Line2D.py
def getY(line,x):
	a=line[0]
	b=line[1]
	c=line[2]

	if b == 0:
		# print (a,b,c)
		if a*x == int (-1*c):
			return  math.inf
		else :
			return None
	return int(-1*round((float(a*x+c)/float(b)),0))

## Use Plane3D.py
def getPlane(pt1,pt2,pt3):

	# General Equation of Vector ax +by+cz +d =0
	vectorA= (pt3[0]-pt1[0],pt3[1]-pt1[1],pt3[2]-pt1[2])
	vectorB= (pt2[0]-pt1[0],pt2[1]-pt1[1],pt2[2]-pt1[2])

	a = vectorA[1]*vectorB[2] -vectorA[2] * vectorB[1]
	b =	vectorA[2]*vectorB[0] -vectorA[0] * vectorB[2]
	c = vectorA[0]*vectorB[1] -vectorA[1] * vectorB[0]

	d = -1*(a*pt1[0] + b*pt1[1] + c*pt1[2])

	return (a,b,c,d)

## USe Plane3D.py
def getZ( plane,x,y):

	val = plane[0]*(x) + plane[1]*(y) +plane[3]
	if (plane[2]) == float(0):
		# print (val)
		# Dealing with edge Case if Z coffiecient is Zero So there is no projection on XY PLane
		return float(0)

	return float(-1* (val/plane[2]))

def getPointInsideTriangle (plane,vertexes):

	points=[] 
	#sort the list basis of x cord
	vertexes = sorted ( vertexes, key=itemgetter(0))

	# print (plane)

	#get equation of Lines
	baseLine = getLine(vertexes[0],vertexes[2])

	for x in range (vertexes[0][0],vertexes[1][0],1):

		newLine = getLine(vertexes[0],vertexes[1])

		y_base = getY(baseLine, x)
		if y_base == math.inf:
			y_base = vertexes[0][1]

		y_new = getY(newLine,x)
		if y_new == math.inf:
			y_new = vertexes[1][1]

		for y in range(min(y_base,y_new),max(y_new,y_base)+1,1):
			# get aa value of Z 
			z = getZ(plane,x,y)		
			points.append((x,y,z))

	for x in range ( vertexes[1][0],vertexes[2][0]+1,1):

		newLine = getLine(vertexes[1],vertexes[2])
		y_base = getY(baseLine, x)
		if y_base == math.inf:
			y_base = vertexes[1][1]
		y_new = getY(newLine,x)
		if y_new == math.inf:
			y_new = vertexes[2][1]

		for y in range(min(y_base,y_new),max(y_new,y_base)+1,1):
			# get aa value of Z 
			z = getZ(plane,x,y)
			points.append((x,y,z))

	return points

## will retain this function 
def createDisplacementMap( Faces, X,Y,Z,l=900,b=1000):

	# Displacement Map for size (x,y)
	displacementMap = np.zeros((l,b) ,dtype = float)

	minX= float(-1) # min(X)
	minY= float(-1) #min(Y)
	minZ= float(-1) #min(Z) 
	maxZ = max(Z)
	#Scale Factor
	scaleX = int ( 900/(max(X) - minX))
	scaleX = int ( b / float(2))
	scaleY = int ( 1000/(max(Y) - minY))
	scaleY = int ( l / float(2))
	scaleZ = int ( 1000/(maxZ - minZ))


	for face in Faces:

		x1= int (round ( (X[face[0]] - minX ) * scaleX,0)) 
		y1= int (round ( (Y[face[0]] - minY ) * scaleY,0))
		z1= int (round ( (Z[face[0]] - minZ ) * scaleZ,0))

		# print(x1,y1)

		x2= int (round ( (X[face[1]] - minX ) * scaleX,0))
		y2= int (round ( (Y[face[1]] - minY ) * scaleY,0))
		z2= int (round ( (Z[face[1]] - minZ ) * scaleZ,0))

		x3= int (round ( (X[face[2]] - minX ) * scaleX,0))
		y3= int (round ( (Y[face[2]] - minY ) * scaleY,0))
		z3= int (round ( (Z[face[2]] - minZ ) * scaleZ,0))

		plane = getPlane((x1,y1,z1),(x2,y2,z2),(x3,y3,z3))
		points = getPointInsideTriangle(plane,[(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)])

		for (x,y,z) in points:
			value = int ( z )
			displacementMap[l-1-y][b-1-x]= max ( value , displacementMap[l-1-y][b-1-x])

	return displacementMap

## will retain this function 
def testDisplacementMap( displacementMap):

	img = np.zeros((displacementMap.shape[0],displacementMap.shape[1]) ,dtype = np.uint8)
	img.fill(255)
	zmax=np.amax(displacementMap)
	# print("zmax in displacementMap",zmax)

	maxI = 200
	minI = 50

	for i in range(displacementMap.shape[0]):
		for j in range(displacementMap.shape[1]):
			if displacementMap[i][j] > 0:
				img[i][j] = minI +int ( float ((maxI-minI)*displacementMap[i][j])/float(1*zmax))
	return img

# Use BresenhamLine.py
def bresenhamLine( img ,vertexes):

	# slope of line
	# Bresnan Line Implementatiom

	#Sort the List on the basis of X cordinate
	vertexes = sorted ( vertexes, key=itemgetter(0))
	x1 = vertexes[0][0]
	y1 = vertexes[0][1]
	x2 = vertexes[1][0]
	y2 = vertexes[1][1]

	## Flag to hold is slope negative 
	slopenegative = True if (y2-y1) < 0 else False

	dx = abs(x2 -x1)
	dy = abs(y2 -y1)
	a= dx
	b= dy

	# Flag hold the value whether  slope is less than 1
	flag = True
	if dy > dx:
		a = dy
		b = dx
		flag =False

	######## Follow the Link for derivation
	######## https://www.tutorialspoint.com/computer_graphics/line_generation_algorithm.htm


	###### Getting the list of all coordinates
	p = 2*b - a
	count = 0

	for i in range (a+1):

		### Slope less than 1
		if flag == True :
			img[ y1 + count][x1+i] = np.uint8(0)

		### slope greater than 1
		else:
			#### slope negative check 
			if slopenegative == False:
				img [y2 - i][x2 - count] = np.uint8(0)
			else:
				img [y2 +i ][x2 - count] = np.uint8(0)


		if p >=0:
			count += int( -1 ) if slopenegative == True else int(1)
			p += (2*b - 2*a)

		else:
			p += (2*b)

	return img

# Use Vector2D.py
def addVector ( A, B):
	return ((A[0]+B[0]),(A[1]+B[1]))

# Use vector2D.py ScaledVector
def getVector (vec, mag):
	vec = getUnitVector(vec)
	return ((mag * vec[0]),(mag*vec[1]))

# Use Vector2D.py 
def getUnitVector(vec):

	if vec[0] == float(0) and vec[1] == float(0):
		return ((float(0),float(0)))

	mag = math.sqrt(vec[0] **2 + vec[1] **2)
	return ((vec[0]/mag,vec[1]/mag))

# will retain this function 
def makeGray ( img , displacementMap):


	l = int(displacementMap.shape[0])
	b = int(displacementMap.shape[1])

	for i in range ( l ):
		for j in range ( b ):
			img[i][j] = np.uint8(200) if displacementMap[i][j] > 0 else np.uint8(255) 

	return img

# Will retian this function
#Return Image
def createArt(displacementMap, scale =0.25):

	print ("Create Art being called")
	scaleA = 0.25
	scaleB = 0.2

	scaleLeft = 1

	l = int(displacementMap.shape[0])
	b = int(displacementMap.shape[1])

	img = np.zeros((l,b) ,dtype = np.uint8)
	img.fill(255)

	# img = testDisplacementMap (displacementMap)
	# img = makeGray (img, displacementMap)

	start = 1

	randomLines = getRandomLinesList(len=l-start -1, start= start, maxInterval = 6)
	# randomLines=[]
	# randomLines.append(int(550))

	print ("length of randomLines",len (randomLines))

	for i in randomLines:
		
		# print (" this is sucker ")
		# print ( "i ", i)
		# img = bresenhamLine(img, [(0,i),(899,i)])

		currY = i
		prevY = currY

		prevX = 0
		currX = 0

		prevZ = None
		currZ = None

		# Previous Logic
		left = 0.0

		# New logic based upon vector addition
		currVec = (float(1), float(0))
		prevVec = (float(1), float(0))


		while currX < b and currY < l and currY >=0:

			# print (currX, currY, currZ)
			# print ( (displacementMap[currY][currX]) )
			# print (currX, (currY)) 
			
			currZ = displacementMap[currY][currX]

			if currX == 0:
				img[currY][currX]=np.uint8(0)
				prevX = currX
				prevY = currY
				prevZ = currZ
				currX+=1
				continue

			#create a line prev to current 
			# temporary solution
			# img[currY][currX]=np.uint8(0)
			img = bresenhamLine(img,[(currX,currY),(prevX,prevY)])

			

			# dzx = scaleA * ((displacementMap[currY][(currX+1)%b] - displacementMap[currY][(currX-1)%b]))
			# dzy = scaleB * (displacementMap[(currY+1)%l][currX] - displacementMap[(currY-1)%l][currX])

			# # print ("dzx,dzy", dzx , dzy)

			# magnitude = scale*(currZ- prevZ)

			# newVec = getVector((dzy,dzx), magnitude)




			####################################
			# Vector Logic to find the value

			# print (newVec)

			# currVec = addVector(currVec,newVec)

			# prevY = currY
			# prevX = currX
			# prevZ = currZ

			# p = 0
			# o = 1 

			# if currVec[p] > abs(currVec[o]):
			# 	currX += 1
			# else:
			# 	if currVec[o] > 0:
			# 		currY += 1
			# 	else:
			# 		currY -=1

			# 	if currVec[p] == abs(currVec[o]):
			# 		currX +=1

			####################################
			###### Old logic 


			#### Need modification
			#### Reason it depends upon the slope of boundary 
			##### If slope is < 1 than damper should be adjusted according
			##### Will work afterwards

			damper = 0.5
			prevY = currY 
			alpha = (scale*(currZ-prevZ))  + scaleLeft*left
			roundAlpha = int (round(alpha,0))
			
			while abs(roundAlpha) > 1:
				alpha = roundAlpha*damper
				roundAlpha = int (round(alpha,0))

			left = alpha - roundAlpha


			currY = currY + roundAlpha

			prevX = currX
			currX += 1

			prevZ = currZ

	
	return img

def Test():

	## Test the vector2D 
	A= vector.vector2D(3,0)
	mod = A.getModulus()
	A = A.unitVector()
	A = A.scaledVector(5)
	A = A.addVector(vector.vector2D(0,3))
	A.printVector()
	print( mod)


	## Testing the vector 3D
	B = vector.vector3D(3,2,0)
	B.printVector()


	# print (addVector((1,2),(1,2)))
	# ##########################################
	# # Testing Vector Addition

	# print (getVector((0,0),10))
	# ############################################
	# #Testing Vector intialitation


	# A = getLine((1,2),(3,2))
	# print (A)
	# ##########################################
	# ## test for getting Equation of line

	# B = getY(A, 4)
	# print ("value of Y x=4 ",B)
	# ##########################################
	# # Testing point y cordinate

	# plane = getPlane((4,0,0),(0,4,0),(0,0,4))
	# print ("plane ", plane)
	# ###########################################
	# # Testing Equation of Plane

	# # Tested All 4 possible Scenarios 
	# points = getPointInsideTriangle(plane,[(0,0),(0,4),(4,0)])
	# print (points)
	# ##########################################
	# # Testing point inside the triangles


	# # Z = getZ(plane,4, 5)
	# # print("Z for plane ", plane ,Z)
	# # #######################################
	# # TESTING THE VOXELIZATION IN XY ORTHOGONAL PLACE

	# displacementMap = createTestDisplacementMap(key ="hemisphere")
	
	# # Test Displacement Map
	# Art = testDisplacementMap(displacementMap)
	# img = Image.fromarray(Art)
	# img.save ('DisplacementMap.png')
	# #######################################

	# #Creating the Art
	# Art = createArt(displacementMap = displacementMap)

	
	# # img = np.zeros((900,900) ,dtype = np.uint8)
	# # img.fill(255)
	# # Art = bresenhamLine(img,[(300,700),(200,600)])
	# ##### Testing bresenham Line

	# # Display the image
	# img = Image.fromarray(Art)
	# img.save ('result.png')
	########################################
	# testing create Art algorithm


if __name__ == '__main__':

	Test()
	
	# Reading the vertices
	# X, Y, Z = readVertexs("vertex_new.txt")
	# print ("Number of Vertexs",len(X))

	# #Reading triangle Vertexs
	# Faces = (readFaces("face_new.txt"))
	# print ("Number of triangular Faces",len(Faces))

	# #get the all possible Edges
	# Edges = getEdges(Faces)
	# print ("Total Number of Edges",len(Edges))

	# #group edges based upon slope
	# #Justified as the input file contains triangle which are non overlapping
	# #Edges are not partialy shared
	# boundaryEdges= getBoundaryEdges(X,Y,Z, Edges)
	# print ("Total Number of Boundary Edges",len(boundaryEdges)) 

	# # print(type(boundaryEdges))
	# ## Rteurn the Closed Surface to the Face
	# # Faces,X,Y,Z= getClosedSurface(Faces,X,Y,Z, boundaryEdges)
	# # print("Total number of Faces ",len(Faces))

	# #######################################################################
	# #working code 

	# displacementMap = createDisplacementMap(Faces= Faces, X=X,Y=Y,Z=Z)

	# print ("Displacemnt Map Shape",displacementMap.shape)

	# # print("max value in displacement",np.amax(displacementMap))

	# Art=testDisplacementMap(displacementMap)
	# img = Image.fromarray(Art)
	# img.save ('DisplacemntMapFace.png')


	# ##########################################################################
	# # Voxelisation of Coordinates in XY plane

	# ## Getting Displacemnt Map 

	# # Trial creating Displacement MAp
	# # displacementMap = createDisplacementMap()
	# # print ( "displacement map shape",(displacementMap.shape))

	# # # #Creating the Art
	# Art = createArt(displacementMap = displacementMap)

	# # # # Display the image
	# img = Image.fromarray(Art)
	# img.save ('finalOutput.png')


	#########################################################################
	# WORKING PEICE OF CODE FOR DISPLACEMENT MAPPING
	# VERFIED FOR CUBE AND WEDGE

	# Test()

	#nextnthing to Do
	
	# 1. Voxelsiser to covert Surafce to list of voxels, ( only in XY Plane)
	# 2. Add BAse Fram to it. (Automatically Dealt)
	# 3. Write an algorithm to make a geodesic (Not required using Displacemnt to serve the purpose)
	# 4. ploting the point in geodesic lines onto plane (Implemented GEodesic Not required )


	# skewEdges=getSkewLines(X,Y,Z, parllelLines)
	# print(len(skewEdges))


	# # Reading Texture File 
	# Texture = (readFileTuple("texture.xslx"))

	# #Extracting the Boundary Points
	# BoundaryPoints = getBoundaryPoints(x, y, z , Faces)
	# # print (len(BoundaryPoints), len (x))

	# #Fittered Faces
	# # FilteredFaces=getFilteredFaces(Faces, BoundaryPoints)
	# # print ( len (FilteredFaces) , len (Faces))
	



	# #creating a flat surface
	# u = np.linspace(-1,1, num =2*1000)
	# v = np.linspace(-1,1, num=2*1000)
	# u,v = np.meshgrid(u,v)
	# u = u.flatten()
	# v = v.flatten()

	# h=np.zeros(2*1000)

	# points2D = np.vstack([u,v]).T
	# tri = Delaunay(points2D)
	# simplices = tri.simplices

	# 	x=np.array(x)
	# 	z=np.array(z)
	# 	y=np.array(y)
	# 	faces=np.array(Faces)

	# # coverting to np arrays
	# 	# x=np.concatenate((np.array(cordX), u), axis=0)
	# 	# z=np.concatenate((np.array(cordZ), h), axis=0)
	# 	# y=np.concatenate((np.array(cordY), v), axis=0)
	# 	# faces=np.concatenate((simplices, faces), axis=0)

	# 	# print ( len(faces), len(cordX))

	# fig1 = FF.create_trisurf(x=X, y=Y, z=Z,
	# 							colormap = [(0.4, 0.15, 0), (1, 0.65, 0.12)],
	# 							# color_func=,
	# 							simplices=Faces,
	# 							title="Mobius Band",
	# 							plot_edges=False,
	# 							show_colorbar=False
	# 							)

	# print (fig1)

	# py.plot(fig1, filename="face.html")

