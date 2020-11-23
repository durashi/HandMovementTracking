import cv2

def add_box(original_image_path, segmented_image_path, num):
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

    print("Number of Contours found = " + str(len(contours))) 

    object_matrix = []

    e_w = int(width / 100)
    e_h = int(height / 100)

    for contour in contours:
    

        extLeft = tuple(contour[contour[:, :, 0].argmin()][0])
        extRight = tuple(contour[contour[:, :, 0].argmax()][0])
        extTop = tuple(contour[contour[:, :, 1].argmin()][0])
        extBot = tuple(contour[contour[:, :, 1].argmax()][0])
        if (extRight[0] - extLeft[0] > 5 * e_w ) and (extBot[1] - extTop[1] > 5 * e_h):
            object_list = [(max(extLeft[0] - e_w, 0), min(extRight[0] + e_w , width)),(max(extTop[1] - e_h, 0), min(extBot[1] + e_h, height))]
            #object_list = [(extLeft[0],extRight[0]), (extTop[1], extBot[1])]
            object_matrix.append(object_list)

    areas = []
    areas_copy = []
    obj_matrix_largest = []
    for i in object_matrix :
        area = (i[0][1] - i[0][0]) * (i[1][1] - i[1][0])
        areas.append(area)
        areas_copy.append(area)

    
    if len(object_matrix) >= 2 :
        max_area = areas.index(max(areas))
        sec_max_area = areas.index(sorted(areas_copy, reverse=True)[1])
        obj_matrix_largest.append(object_matrix[max_area])
        obj_matrix_largest.append(object_matrix[sec_max_area])

    elif len(object_matrix) == 1 : 
        max_area = areas.index(max(areas))
        obj_matrix_largest.append(object_matrix[max_area])

    if len(obj_matrix_largest) > 0 :
        for i in obj_matrix_largest:
            original_image = cv2.rectangle(original_image, (i[0][0], i[1][0]), (i[0][1], i[1][1]), (0, 0, 0), 3) 


    cv2.imwrite('box_images/{}.png'.format(num), original_image)



original_path = 'images_original/'
segmented_path = 'images/'

for i in range(1, 721):
    original = original_path + str(i) + '.png'
    segmented = segmented_path + str(i) + '.png'
    add_box(original, segmented, i)


cv2.destroyAllWindows()
