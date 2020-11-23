import cv2


segmented_image = cv2.imread('images/89.png')
original_image = cv2.imread('images_original/89.png')

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

print("Number of Contours found = " + str(len(contours))) 

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


print(object_matrix)

j = 1

for i in object_matrix:
    original_frame = original_image[i[1][0]:i[1][1], i[0][0]:i[0][1]] 
    segmented_frame = segmented_image[i[1][0]:i[1][1], i[0][0]:i[0][1]] 
    cv2.imshow('crop_frame', original_frame)
    cv2.waitKey(3000)
    cv2.imshow('segmented_frame', segmented_frame)
    cv2.waitKey(3000)

    cv2.imwrite('test_images/{}.png'.format(str(j)), original_frame)
    j += 1

cv2.destroyAllWindows()

