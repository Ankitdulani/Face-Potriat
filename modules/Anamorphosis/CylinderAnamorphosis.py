from  modules.Utils import vector
from modules.Shapes import Cylinder
from modules.Geometry import Line
from modules.Geometry import Plane
from modules.Utils import IOImage
from modules.Utils import BresenhamAlgo

from PIL import Image
import numpy as np
import math 
import cv2

"""
Inputs:
	Image,
	Observer Position,
	Radius of the cylinder,
	Size of Art 
	
Assumptions:
	Location of the Cylindrical Mirror will be by defalt (b/2, l)
	Plane of Image is Perpendicular to the Plan of Projection and Location is fixed adjusted according to the mirro center
	not curvilinear

"""
class Anamorrphosis:
	def create(name):
		# on safer side
		l = 2000
		b = 2000

		Reader = IOImage.reader()
		Writer = IOImage.writer()
		imgL = int(1*1024)
		imgW = int(	1*768)
		image_size = (120,160)
		# reades black and white image

		img  = Reader.readImage(str(name),"",".jpg","anomorphosis" )
		img = cv2.resize( img , ( imgW , imgL ))
		img = cv2.rotate(img, cv2.ROTATE_180)
		Writer.writeImage("CoolDude.jpg",img,folder="Text")

		output = np.zeros((b,l,3) ,dtype = np.uint8)
		output.fill(255)

		# Cylinderical 
		# Radius on the basi keeping view angle limited to 120 degree
		radii = int ( image_size[0] *0.5 *1.1 / math.cos( math.pi / 4 ) )
		radii = 200 
		print("radii: ",radii)


		ccenter = vector.vector3D( b/2 , l/2 , 0)
		axis = vector.vector3D(0,0,1)
		cylindrical_mirror = Cylinder.Cylinder.newCylinder( radii, ccenter, axis)


		output = cv2.circle(output,(int(ccenter.x) , int(ccenter.y)),radii,(0,0,0),3)
		Writer.writeImage("circleTest.jpg",output,folder="anamorphosis")

		image_loc = vector.vector3D( int (-1*image_size[0]/2 ),
									  int (1*radii/2),
									  0)
		#output = cv2.circle(output,(int(image_loc.x) , int(image_loc.y)),10,(255,0,0),3)
		image_loc = vector.vector3D.addVector(image_loc,ccenter)
		# output = cv2.circle(output,(int(image_loc.x) , int(image_loc.y)),10,(255,0,0),3)
		#ouput plane
		output_plane = Plane.Plane3D()
		output_plane.setAxis(vector.vector3D(0,0,1))
             
		# Observer
		Observer = vector.vector3D( b/2 , l-100 , 200 )
		output = cv2.circle(output,(int(Observer.x) , int(Observer.y)),10,(0,255,0),3)
		# print("\nObserver")
		# Observer.printVector()

		di = float(image_size[0]) / float(imgW)
		dj = float(image_size[1]) / float(imgL)

		maxX = 787987
		maxY =0

		percount = 0

		# AxisLine = Line.Line3D.newLine(cylindrical_mirror.pos,cylindrical_mirror.Axis)
		print ("processing")


		# Algorithm based upon vector and resolution.
		for  i in np.arange(0,image_size[0], di):
			val = i*100/(di * imgW)
			if val >= percount:
				print("{:.2f}".format(val)," %")
				percount+=5
			if val >=100:
					break

			for j in np.arange( 0,image_size[1],dj):
				Image_Point = vector.vector3D.addVector(image_loc,vector.vector3D(i,-1*j,0))
			
				output[int(Image_Point.y),int(Image_Point.x),:] = img[ int(j/dj),int(i/di) ,:]

				incidentRay = Line.Line3D.newLine( Observer, vector.vector3D.subVector(Image_Point,Observer) )

				intersectionPoint = cylindrical_mirror.getIntersectionWithLine(incidentRay)
				if intersectionPoint == None :
					reflectedRay = intersectionPoint
				else :
					normalVec,output = cylindrical_mirror.getNoramlVec(intersectionPoint,output)
					reflectedRay = Line.Line3D.newLine ( intersectionPoint , vector.vector3D.getReflectedVector(normalVec,incidentRay.dir))
	
				pt = output_plane.getIntersectionWithLine(reflectedRay)

				output[int(pt.y),int(pt.x),:] = img[ int(j/dj),int(i/di) ,:]
				
		fileName = "Art_"+str(name)+".jpg"
		Writer.writeImage(fileName,output,folder="anamorphosis")
			
	def create_using_diff( name):

		l = 2000
		b = 2000

		Reader = IOImage.reader()
		Writer = IOImage.writer()

		image_size = (400,600)

		img  = Reader.readImage(str(name),"",".jpg","anomorphosis" )
		img = cv2.resize( img , image_size)

		output = np.zeros((b,l,3) ,dtype = np.uint8)
		output.fill(255)

		radii = 200 
		# assuming circle situated at the center of the images 
		ccenter = vector.vector3D( b/2 , l/2 -200 , 0)
		output = cv2.circle(output,(int(ccenter.x) ,int(ccenter.y)),int(radii),(0,0,0),3)

		axis = vector.vector3D(0,0,1)
		cylindrical_mirror = Cylinder.Cylinder.newCylinder( radii, ccenter, axis)

		# define image countour
		image_loc = vector.vector3D(int(-1*radii),int(0 -image_size[1]),0)
		# output = cv2.circle(output,(int(image_loc.x+b/2) , int(image_loc.y+l/2)),10,(0,0,0),3)

		# Observer
		Observer = vector.vector3D(0 , l/2 - 100 , 0)
		# output = cv2.circle(output,(int(Observer.x+b/2) , int(Observer.y+l/2)),10,(255,0,0),3)

		u_observer = vector.vector3D.getUnitVector(Observer)

		q_allowed = math.acos( radii  / Observer.getMagnitude())

		percount = 0
		for i in np.arange(-b/2,b/2,1):

			val = (i+b/2)*100/b
			# 	print(  val )
			if val >= percount:
				print("{:.2f}".format(val)," %")
				percount+=5
			if val >=100:
					break

			for j in np.arange(0,l/2,1):

				image_point = vector.vector3D(i,j,0)
			
				# checking the point is not inside the circle
				if (image_point.x**2 + image_point.y**2 <= radii**2+1):
					continue

				# output = cv2.circle(output,(int(image_point.x+b/2) , int(image_point.y+l/2)),10,(0,255,0),3)

				u_image_point = vector.vector3D.getUnitVector(image_point)

				u_normal_vec = vector.vector3D.addVector(u_image_point,u_observer) 
				intersection_point = vector.vector3D.scaledVector(u_normal_vec, radii)	
				input_point = vector.vector3D()
				# # validate whether the point of intersection is valid
				# # Angle should be more than the allowed angle 
				if abs( vector.vector3D.getAngle( Observer, intersection_point ) ) <= q_allowed:
					vec = vector.vector3D.subVector(image_point,intersection_point)
					ref_vec = vector.vector3D.getReflectedVector(u_normal_vec,vec)
					input_point = vector.vector3D.addVector(intersection_point,ref_vec)
				else:
					# input_point = image_point 
					continue
		
				# check whether the point inside the countour
				# vector positon and the value are less than the length and the breadth
				pt = vector.vector3D.subVector(input_point , image_loc).vecToIntegerPoint()

				# output = cv2.circle(output,(int(pt.x) +b/2, int(pt.y))+l/2,10,(255,0,0),3)
				if (pt.x >= 0 and pt.x < image_size[0]) and (pt.y >= 0 and pt.y < image_size[1]):
					pt_r = vector.vector3D.addVector(ccenter,image_point).vecToIntegerPoint()
					output[pt_r.y,pt_r.x,:] = img[ pt.y,pt.x,:]


		Writer.writeImage("test.jpg",output,folder="anamorphosis")





