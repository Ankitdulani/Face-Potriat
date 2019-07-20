import sys
import cv2
import os.path
import numpy as np
import random
from pathlib import Path
from modules import BresenhamAlgo
from modules import IOImage
from modules import imageProcessing as processor

# Return Random list of y corrdinated for lines
def getRandomLinesList( len,start=1,maxInterval=10):

	randomLines=[]

	# randomLines.append(256)
	# randomLines.append(258)
	# return randomLines

	x = start
	maxWidth =5 
	minWidth = 3
	while x < len + start - maxWidth:
		width = random.randrange(minWidth,maxWidth,1)
		randomLines.append(x)
		randomLines.append(x+width)
		x += width
		x+= random.randrange(2,maxInterval,1)

	return randomLines

def createArt(depth_map, img , scale =0.25 , randomLines = [] ):

	print ("Create Art being called")

	isColor = False
	if len(img.shape) >= 3:
		isColor = True

	scaleA = 0.25
	scaleB = 0.2

	scaleLeft = 1

	l = int(depth_map.shape[0])
	b = int(depth_map.shape[1])

	previousY = np.zeros( b ,dtype = int)

	start = 1
	count = 0

	if len(randomLines) == 0:
		randomLines = getRandomLinesList(len=l-start -1, start= start, maxInterval =15)

	print ("length of randomLines",len (randomLines))

	for i in range(len(randomLines)):

		currY = randomLines[i]
		prevY = currY

		prevX = 0
		currX = 0

		prevZ = None
		currZ = None

		left = 0.0

		if i % 2 ==0:
			previousY.fill(0)
			count +=1

		while currX < b:
			
			currZ = float(depth_map[currY][currX])

			if currX == 0:
				img[currY][currX]=np.uint8(0)
				prevX = currX
				prevY = currY
				prevZ = currZ
				currX+=1
				

			if i%2 == 0:
				previousY[currX]=currY
			else:
				img = BresenhamAlgo.bresenhamLine.drawLine(img,[(currX,currY),(currX,previousY[currX])])

			if currX == 0:
				continue

			#create a line prev to current 
			# img[currY][currX]=np.uint8(0)
			
			#### Need modification
			#### Reason it depends upon the slope of boundary 
			##### If slope is < 1 than damper should be adjusted according
			##### Will work afterwards

			damper = 0.5
			prevY = currY 
			alpha = (scale*(currZ-prevZ))  + scaleLeft*left
			roundAlpha = int (round(alpha,0))

			# print(roundAlpha)
			
			while abs(roundAlpha) > 1:
				alpha = roundAlpha*damper
				roundAlpha = int (round(alpha,0))

			left = alpha - roundAlpha

			currY = currY + roundAlpha
			if currY >= l:
				currY =l-1

			prevX = currX
			currX += 1

			prevZ = currZ

		
	print("intialised to zero",count)
	return img


if __name__ == '__main__':

	Writer = IOImage.writer()

	name_tag=sys.argv[1] 
	path = Path(os.getcwd())
	path = Path('/Users/ankitdulani/Documents/personal/git/Face-Potriat')
	resource_path =  path / 'resources'
	result_path = path / 'result'

	# loading the depth image 
	fileName = "depth_" + name_tag +".jpg"
	filePath = resource_path / fileName
	depth_map = cv2.imread(str(filePath),cv2.COLOR_BGR2GRAY)
	depth_map = depth_map[:,:]

	# loading the base image 
	fileName = "img_" +name_tag +".jpg"
	filePath = resource_path /fileName
	img = cv2.imread(str(filePath))

	# cv2.imshow('colored image',img)

	grayScaleImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	Writer.writeImage('Gray_1_0.png',grayScaleImage)

	Sketch = processor.IP.createSketch(img)
	Writer.writeImage('SketchColor_1_0.png',Sketch)


	adjusted = processor.IP.createBlackWhite(image = img, gamma = 2)
	Writer.writeImage('AdjustedGame.png',adjusted)

	l = int(depth_map.shape[0])
	b = int(depth_map.shape[1])
	start = 1
	randomLines = getRandomLinesList(len=l-start -1, start= start, maxInterval =10)

	blank_img = np.zeros((512,512),dtype=np.uint8)


	Art = createArt(depth_map = depth_map, img = adjusted , randomLines = randomLines)
	Writer.writeImage('FacePotrait_yami.png',Art)

	






