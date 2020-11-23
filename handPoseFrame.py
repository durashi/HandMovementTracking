from __future__ import division
import cv2
import time
import numpy as np

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

#original_image = cv2.imread('images_original/89.png')



def add_keypoints(original_image_path, segmented_image_path, num):
    segmented_image = cv2.imread(segmented_image_path)
    original_image = cv2.imread(original_image_path)

    height = original_image.shape[0]
    width = original_image.shape[1]


    segmented_image = cv2.resize(segmented_image, (int(width), int(height)), interpolation = cv2.INTER_LINEAR) 

    row_current = 0
    object_ = {
        'x_start': -1,
        'x_end' : -1,
        'y_start' : -1,
        'y_end': -1
    }

    # Grayscale 
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY) 
    
    # Find Canny edges 
    edged = cv2.Canny(gray, 30, 200) 
    cv2.waitKey(0) 
    
    # Finding Contours 
    # Use a copy of the image e.g. edged.copy() 
    # since findContours alters the image 
    contours, hierarchy = cv2.findContours(edged,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 


    object_matrix = []

    e_w = int(width / 100)
    e_h = int(height / 100)

    for contour in contours:
    

        extLeft = tuple(contour[contour[:, :, 0].argmin()][0])
        extRight = tuple(contour[contour[:, :, 0].argmax()][0])
        extTop = tuple(contour[contour[:, :, 1].argmin()][0])
        extBot = tuple(contour[contour[:, :, 1].argmax()][0])
        if (extRight[0] - extLeft[0] > 2 * e_w ) and (extBot[1] - extTop[1] > 2 * e_h):
            object_list = [(max(extLeft[0] - e_w, 0), min(extRight[0] + e_w , width)),(max(extTop[1] - e_h, 0), min(extBot[1] + e_h, height))]
            #object_list = [(extLeft[0],extRight[0]), (extTop[1], extBot[1])]
            object_matrix.append(object_list)



    keypoint_image = np.copy(original_image)



    for k in object_matrix:
        original_frame = original_image[k[1][0]:k[1][1], k[0][0]:k[0][1]] 
        segmented_frame = segmented_image[k[1][0]:k[1][1], k[0][0]:k[0][1]] 


        frame = original_frame

        frameCopy = np.copy(frame)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        aspect_ratio = frameWidth/frameHeight

        threshold = 0.085

        t = time.time()
        # input image dimensions for the network
        inHeight = 368
        inWidth = int(((aspect_ratio*inHeight)*8)//8)
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inpBlob)

        output = net.forward()


        # Empty list to store the detected keypoints
        points = []

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (frameWidth, frameHeight))

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > threshold :
                cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(keypoint_image, (int(point[0]) + k[0][0], int(point[1] + k[1][0])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(point[0]), int(point[1])))
            else :
                points.append(None)


        for i in range(len(points)) :
            point = points[i]
            if point :
                lst = list(point)
                lst[0] += k[0][0]
                lst[1] += k[1][0]
                point = tuple(lst)

                points[i] = point


        # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(original_image, points[partA], points[partB], (0, 255, 255), 2)
                cv2.circle(original_image, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(original_image, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.imwrite('skeleton/{}.png'.format(str(num)), original_image)
    cv2.imwrite('keypoints/{}.png'.format(str(num)), keypoint_image)


original_path = 'images_original/'
segmented_path = 'images/'

for i in range(1, 721):
    original = original_path + str(i) + '.png'
    segmented = segmented_path + str(i) + '.png'
    add_keypoints(original, segmented, i)
    


print('done')