from modules.Utils import vector
import math
import numpy as np 

class model:
	
	#return the rotation matrix
	def getRotationMatrix(A):

		#Rotation in Axis not workin g properly adding an addition rotation in 
		(a,b,c) = vector.vector3D.getRotaionAngles(A)
		print(a,b,c)

		if (a == c  and a == math.pi/2):
			return (model.getAxisRotationMatrix(-1*math.pi/2,"x"))
		r = model.getAxisRotationMatrix(a,"z").dot(model.getAxisRotationMatrix(c,"y"))
		
		return r
	# return roatation matrix particular to the angel
	def getAxisRotationMatrix(q,axis):
		ca =round(math.cos(q),8)
		sa = round(math.sin(q),8)
		r = np.zeros((4,4))
		r[3][3] = 1

		if axis is "x":
			r[1][1] = ca
			r[2][2] = ca
			r[1][2] = -1*sa
			r[2][1] = 1*sa
			r[0][0] = 1
		elif axis is "y":
			r[0][0] = ca
			r[2][2] = ca
			r[0][2] = sa
			r[2][0] = -1*sa
			r[1][1] = 1
		elif axis is "z":
			r[0][0] = ca
			r[1][1] = ca
			r[1][0] = sa
			r[0][1] = -1*sa
			r[2][2] = 1

		return r