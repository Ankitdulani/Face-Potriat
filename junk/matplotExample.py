from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np



import plotly.plotly as py
import plotly.figure_factory as FF
import plotly.graph_objs as go


path=("vertex.xslx")
file =open(path,'r')

cordX=[]
cordY=[]
cordZ=[]

for line in file:
	x=line.strip().split(',')
	cordX.append(float(x[0]))
	cordY.append(float(x[1]))
	cordZ.append(float(x[2]))

n_radii = 8
n_angles = 36

# Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
radii = np.linspace(0.125, 1.0, n_radii)
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

# Repeat all angles for each radius.
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)

# Convert polar (radii, angles) coords to cartesian (x, y) coords.
# (0, 0) is manually added at this stage,  so there will be no duplicate
# points in the (x, y) plane.
x = np.append(0, (radii*np.cos(angles)).flatten())
y = np.append(0, (radii*np.sin(angles)).flatten())

# Compute z to make the pringle surface.
z = np.sin(-x*y)

fig = plt.figure()
ax = fig.gca(projection='3d')

print (cordX[0:100])
print ((cordY[0:100]))
ax.plot_trisurf(cordX, cordY, cordZ, linewidth=0.2, antialiased=True)

plt.show()