import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from infer_segmentation import Infer

cap = cv2.VideoCapture('V2.mp4')

classes_dict = {'background': 0,'hand': 1}
classes_to_train = ['hand']


gtf = Infer()

gtf.Data_Params(classes_dict, classes_to_train, image_shape=[716,1024])

gtf.Model_Params(model="Unet", backbone="efficientnetb3", path_to_model='seg_hand_trained/best_model.h5')

gtf.Setup()



#video = cv2.VideoWriter('video.avi',-1,1,(int(width/2) ,int(height/2)))
count = 1
# load the video
while(cap.isOpened()):                    # play the video by reading frame by frame
    ret, frame = cap.read()
    if ret==True:
        # get dimensions of image

        dimensions = frame.shape
        
        # height, width, number of channels in image
        height = frame.shape[0]
        width = frame.shape[1]
        channels = frame.shape[2]


        cv2.imwrite('frame.png', frame)
        cv2.imwrite('images_original/{}.png'.format(str(count)), frame)

        pr_mask, img_list = gtf.Predict('frame.png', vis = False)

        hand = [list(map(int,i)) for i in  img_list[1]]
        hand = [[j*255 for j in i] for i in hand]
        hand = np.array(hand)


        img_hand = hand.astype(np.uint8)
        #img_hand = cv2.resize(img_hand, (int(width/2), int(height/2)), interpolation = cv2.INTER_LINEAR) 

        # slow the video using the waitKey (100ms)
        #
        cv2.imwrite('images/{}.png'.format(str(count)), img_hand)
        count += 1

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

