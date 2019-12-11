
import sys

from modules.LineWrapper import LineWrapper

if __name__ == '__main__':

	# Test()

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


