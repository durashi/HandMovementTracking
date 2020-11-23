import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from infer_segmentation import Infer

img = cv2.imread('image.png')
dimensions = img.shape

# height, width, number of channels in image
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]

print(height, width)

gtf = Infer()

classes_dict = {'background': 0,'hand': 1}

classes_to_train = ['hand']

gtf.Data_Params(classes_dict, classes_to_train, image_shape=[716,1024])

gtf.Model_Params(model="Unet", backbone="efficientnetb3", path_to_model='seg_hand_trained/best_model.h5')

gtf.Setup()

pr_mask, img_list = gtf.Predict('image.png', vis = False)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hand = [list(map(int,i)) for i in  img_list[1]]



#print(gray)

name = 'segmented'
image = hand
plt.xticks([])
plt.yticks([])
plt.title(' '.join(name.split('_')).title())
plt.imshow(image)
plt.show()

hand = [[j*255 for j in i] for i in hand]
hand = np.array(hand)
print(hand)

img_hand = hand.astype(np.uint8)

img_hand = cv2.resize(img_hand, (int(width/2), int(height/2)), interpolation = cv2.INTER_LINEAR) 

cv2.imshow('frame', img) 
cv2.waitKey(3000) 
cv2.imshow('gray', gray)  
cv2.waitKey(3000) 
cv2.imshow('gray', img_hand)  
cv2.waitKey(3000)

cv2.destroyAllWindows()