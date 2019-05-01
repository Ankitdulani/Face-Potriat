import plotly
import plotly.offline as py
import plotly.figure_factory as FF
import plotly.graph_objs as go

import numpy as np
from scipy.spatial import Delaunay 


if __name__ == '__main__':


		#creating a flat surface
	u = np.linspace(-1,1, num =1*100)
	v = np.linspace(-1,1, num=1*100)
	u,v = np.meshgrid(u,v)
	u = u.flatten()
	v = v.flatten()

	print (len (u))

	points2D = np.vstack([u,v]).T
	tri = Delaunay(points2D)
	simplices = tri.simplices


	 
	# z=np.zeros(10000,)
	# print ((z))

	fig1 = FF.create_trisurf(x=u, y=v, z=z,
						colormap = [(0.4, 0.15, 0), (1, 0.65, 0.12)],
						simplices=simplices,
						title="Gandu",
						plot_edges=False,
						show_colorbar=False
					)

	py.plot(fig1, filename="Trial.html")