
class Anamorrphosis:
	def create():

		l = 4000
		b = 3000

		# Cylinder Params
		# Cylinder height doesn't matter 
		radii = 100 

		Observer = vector.vector3D(1000,-500,1000)

		Cylinder_Center = vector.vector3D(b/2,l,0)

		Image_Location = vector.subVector(Cylinder_Center,vector.Vector3D(0,20,0))

		for  i in range(0,80,1):
			for j in range (0,120,1):

				Image_Point = vector.addVector(Image_Location,vector.vector3D(i,0,j))

				# Incident Ray Vector 
				Incident_Ray = vector.subVector(Image_Point,Observer)

				# Now find the intersectioon point of Cyclinder and the Incident Ray



		"""
		Inputs:
			Image,
			Observer Position,
			Radius of the cylinder,
			Size of Art 
			
		Assumptions:
			Location of the Cylindrical Mirror will be by defalt (b/2, l)
			Plane of Image is Perpendicular to the Plan of Projection and Location is fixed adjusted according to the mirro center

		Mathematics:

			Normal is radiusin

		"""
