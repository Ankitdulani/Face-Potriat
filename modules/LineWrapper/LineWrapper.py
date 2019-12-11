import plotly
import plotly.offline as py
import plotly.figure_factory as FF
import plotly.graph_objs as go


from operator import itemgetter
from scipy.spatial import Delaunay

from modules.Utils import ColorSchema
from modules.Utils import imageProcessing as processor
from modules.LineWrapper import LineWrapperImpl
from modules.LineWrapper import LineWrapperHelper
from modules.Utils import IOImage

import cv2
import numpy as np

class Text:
	def create(name,art_type,minWidth,maxWidth,maxInterval,colorPlate):

		reader = IOImage.reader()
		Writer = IOImage.writer()
		Brush = LineWrapperImpl.LineWrapperImpl()

		img = reader.readImage(name,img_prefix="",folder="Text")
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		disparityMap = cv2.resize(img, (1000,1000))
		disparityMap = 255 - disparityMap

		disparityMap=processor.IP.createBlackWhite(disparityMap)
		Writer.writeImage("test.jpg",disparityMap,folder="Text")

		img = cv2.Canny(disparityMap,10,80)
		baseImage = 255-img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# baseImage = np.zeros((1000,1000), dtype=np.uint8)
		# baseImage.fill(255)

		fileName = "Art_"+str(name)+".jpg"
		Art = Brush.draw(displacementMap = disparityMap, img = baseImage, art_type=art_type,maxInterval=int(maxInterval),colorPlateValue=colorPlate, maxWidth = int(maxWidth), minWidth = int(minWidth))
		Writer.writeImage(fileName,Art,folder="Text")

class Obj:
	def create(name,art_type,minWidth,maxWidth,maxInterval,colorPlate,background = "canny"):

		helper = LineWrapperHelper.Helper()
		Brush = LineWrapperImpl.LineWrapperImpl()

		print("Creating Art of OBJ files")
		Writer = IOImage.writer()

		# print(colorPlate)
		disparityMap, colorImage, LightMap , shadowMap = helper.getInputs("obj",name)

		img = helper.getDisparityImage(shadowMap)
		Writer.writeImage("shadowMap.jpg",img,folder=name)


		baseImage = np.zeros(colorImage.shape,dtype=np.uint8)
		baseImage.fill(255)
		if background == "Dis":
			baseImage = helper.getDisparityImage(disparityMap)
			Writer.writeImage("baseImage.jpg",baseImage,folder=name)

		if background == "color":
			baseImage = colorImage

		if background =="canny":
			print("Using canny")
			disparityImage = helper.getDisparityImage(disparityMap)
			img = cv2.Canny(disparityImage,10,80)
			baseImage = 255-img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		baseImage = processor.IP.overlay(baseImage,ColorSchema.ColorPlate(colorPlate).backgroundColor)
		Writer.writeImage("baseImage_canny.jpg",baseImage,folder=name)

		# colorImageLight = processor.IP.addIntensityMap(colorImage, LightMap)
		# Writer.writeImage('color_intensity.png',colorImageLight,folder=name)

		Art =  Brush.draw(disparityMap,baseImage,art_type=art_type,maxInterval=int(maxInterval),colorPlateValue=colorPlate, maxWidth = int(maxWidth), minWidth = int(minWidth))
		Writer.writeImage("Art_OBJ.jpg",Art,folder=name)

		shifted_image = processor.IP.BlueGreen(Art,10,0)
		Writer.writeImage('Shifted_Hue.png',shifted_image,folder=name)

class Potrait:

	def renderHtml(surface):
		fig1 = FF.create_trisurf(x=surface.X, y=surface.Y, z=surface.Z,
									# color_func = surface.Texture,
									colormap = [(0.4, 0.15, 0), (1, 0.65, 0.12)],
									simplices=surface.Faces,
									title="Mobius Band",
									plot_edges=False,
									show_colorbar=False
									)
		py.plot(fig1, filename="face.html")

	def create(name,art_type,depth,minWidth,maxWidth,maxInterval,colorPlate):

		Reader = IOImage.reader()
		Writer = IOImage.writer()
		helper = LineWrapperHelper.Helper.getInstance()
		Brush = LineWrapperImpl.LineWrapperImpl()
		print(depth)

		img = Reader.readImage(name,folder="base_face")
		resized_image = cv2.resize(img,(1000,1000))

		disparityMap, colorImage, IntensityMap ,shadowMap= helper.getInputs("face",name,depth=depth,baseImage=None)

		adjusted = processor.IP.createBlackWhite(image = resized_image, gamma = 1.5)
		Writer.writeImage('Adjusted_Image.png',adjusted,folder=name)

		ColorBase = processor.IP.overlay(adjusted,ColorSchema.ColorPlate(colorPlate).backgroundColor)
		Writer.writeImage('Background_Image.png',ColorBase,folder=name)

		shifted_image = processor.IP.BlueGreen(adjusted,15)
		Writer.writeImage('Shifted_Hue.png',shifted_image,folder=name)

		(l,b) = disparityMap.shape
		start = 1
		randomLines = helper.getRandomLinesList(len=l-start -1, start= start,  maxInterval=int(maxInterval), maxWidth = int(maxWidth), minWidth = int(minWidth))

		Art = Brush.draw(displacementMap = disparityMap, img = adjusted , randomLines = randomLines,art_type=art_type)
		Writer.writeImage('FacePotrait_adjusted.png',Art,folder=name)


		Art = Brush.draw(displacementMap = disparityMap, img = shifted_image, randomLines = randomLines,art_type=art_type)
		Writer.writeImage('FacePotrait_shifted.png',Art,folder=name)

		Art = Brush.draw(displacementMap = disparityMap, img = ColorBase, randomLines = randomLines,art_type=art_type,colorPlateValue=colorPlate)
		Writer.writeImage('FacePotrait_colored.png',Art,folder=name)

		#renderHtml(surface)

