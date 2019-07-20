import numpy as np
import cv2
import os.path
from pathlib import Path
import matplotlib.pyplot as plt


path = Path('/Users/ankitdulani/Documents/personal/git')

resized_image = cv2.imread('/Users/ankitdulani/Documents/personal/git/depth_net.jpg')
print(resized_image[:,-1,2])



resized_image[:,:,0] = resized_image[:,:,0].transpose()
resized_image[:,:,1] = resized_image[:,:,1].transpose()
resized_image[:,:,2] = resized_image[:,:,2].transpose()

print(resized_image[:,-1,2])

cv2.imwrite( './yami.jpg', resized_image);

# img_h , img_w, img_c  = img_raw.shape
# output_size = 512 

# img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
# hface_cascade = cv2.CascadeClassifier('/Users/ankitdulani/Documents/personal/git/opencv/data/haarcascades/haarcascade_frontalface_default.xml') 
# faces_rects = hface_cascade.detectMultiScale(img_gray, scaleFactor = 1.2, minNeighbors = 5)

# for (x,y,w,h) in faces_rects:

# 	x_new = max(1,min(img_h,x-int(h*.1)))
# 	y_new = max(1,min(img_w,y-int(w*.1)))
# 	h_new = max(1,min(img_h,int(h*1.3)))
# 	w_new = max(1,min(img_h,int(w*1.3)))

# 	cropped_image = img_raw[y_new:y_new+h_new, x_new:x_new+w_new]
# 	cv2.imwrite( './yami_cropped.jpg', cropped_image);
# 	print(cropped_image.shape)

# 	resized_image = cv2.resize(cropped_image,(512,512))

# 	# print(resized_image.shape)


# 	resized_image[:,:,0] = resized_image[:,:,0].transpose()
# 	resized_image[:,:,1] = resized_image[:,:,1].transpose()
# 	resized_image[:,:,2] = resized_image[:,:,2].transpose()

# 	print(resized_image[:,-1,2])

# 	cv2.imwrite( './yami.jpg', resized_image);

# print('Faces found: ', len(faces_rects))


