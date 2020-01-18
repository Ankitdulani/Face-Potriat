
import sys
from modules.LineWrapper import LineWrapper
from modules.Anamorphosis import CylinderAnamorphosis


from modules.Shapes import Cylinder
from modules.Geometry import Line
from modules.Utils import vector
from modules.Geometry import Plane
from modules.Utils import IOImage
from modules.Utils import imageProcessing as processor


def Sketcher( name , val = 6):
	print("This funciton id clalled")
	Reader = IOImage.reader()
	Writer = IOImage.writer()

	img = Reader.readImage(name,folder="base_face")
	grudge = Reader.readImage("grudge%s"%val,img_prefix="",folder="base_face")

	img = processor.IP.createSketch(img,grudge,Writer)
	Writer.writeImage("Sketch_%s.jpg"%name,img,folder="Sketcher")


def Test(name,val):
	c = Cylinder.Cylinder()
	c.setRadius(5)
	c.setPosition(vector.vector3D(0,0,0))

	ln = Line.Line3D.line3D(vector.vector3D(-1,0,6),vector.vector3D(6,0,-1))

	vecA, vecB = Line.Line3D.getMinimumDisPoints(ln,Line.Line3D.newLine(c.pos,c.Axis))
	vecA.printVector()
	vecB.printVector()

	print("Cylinder with line")
	vec = c.getIntersectionWithLine(ln)
	vec.printVector()
	print("Validation",Line.Line3D.getDistance(Line.Line3D.line3D(c.pos, c.Axis),vec))

	eq1 = Line.Line2D.newLine(1, 0, 5)
	eq2 = Line.Line2D.newLine(0, 1, 5 )
	t , u = Line.Line2D.equate( eq1 ,eq2)
	print(t,u)

	print("Plane Test")
	pl = Plane.Plane3D()
	pl.setAxis(vector.vector3D(0,0,1))
	vec = pl.getIntersectionWithLine(ln)
	vec.printVector()

if __name__ == '__main__':

	#Lamborghini_Aventador, Jess_Casual_Walking_001
	name = sys.argv[1]
	extension = sys.argv[2]
	art_type = sys.argv[3] if len(sys.argv) > 3 else 'single_line'
	minWidth = sys.argv[4] if len(sys.argv) > 4 else 1
	maxWidth = sys.argv[5] if len(sys.argv) > 5 else 3
	maxInterval = sys.argv[6] if len(sys.argv) > 6 else 10
	depth =sys.argv[7] if len(sys.argv) >  7 else 700
	colorPlates = sys.argv[8] if len(sys.argv) > 8 else 0

	if (str(extension) == "obj"):
		LineWrapper.Obj.create(name,art_type,minWidth,maxWidth,maxInterval,colorPlates)
	elif (str(extension) == "text"):
		LineWrapper.Text.create(name,art_type,minWidth,maxWidth,maxInterval,colorPlates)
	elif (str(extension) == "face") :
		LineWrapper.Potrait.create(name,art_type,depth,minWidth,maxWidth,maxInterval,colorPlates)
	elif ( str(extension) == "anamorphosis" ):
		CylinderAnamorphosis.Anamorrphosis.create_using_diff(name)	
	elif( str(extension) == "sketch"):
		Sketcher(name,art_type)



