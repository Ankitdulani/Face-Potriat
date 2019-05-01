import xlrd

path=("vertex.xslx")
file =open(path,'r')

vertexs=[]
vertex={}

maxX=-1
maxY=-1
maxZ=-1
minX=1
minZ=1
minY=1

for line in file:

	x=line.strip().split(',')
	vertex["x"]=float(x[0])
	vertex["y"]=float(x[1])
	vertex["z"]=float(x[2])
	vertexs.append(vertex)

	maxX=max(maxX,float(x[0]))
	maxY=max(maxY,float(x[1]))
	maxZ=max(maxZ,float(x[2]))
	minX=min(minX,float(x[0]))
	minY=min(minY,float(x[1]))
	minZ=min(minZ,float(x[2]))

#print(vertexs)

print(maxX, maxY, maxZ)
print(minX,minY,minZ)
#getting min and max of vertexs

for vertex in vertexs:
	vertex["x"] +=minX
	vertex["y"] +=minY
	vertex["z"] +=minZ



maxX +=minZ
maxY +=minY
maxZ +=minZ

# min and max value are same now:	