import cv2
import numpy as np
import glob
 
img_array = []
path = 'box_images/'
for i in range(1, 710):

    filename = path + str(i) + '.png'
    print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('box_frames_2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


