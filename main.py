import cv2
import os

import opencv
import numpy as np

image_dir = "./images/"

files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
files = files[:]

pipeline = opencv.GripPipeline()

def show_image(image):
    cv2.imshow("image", image)
    os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''') 
    cv2.waitKey(0)

for file in files:
    print(file)
    img = cv2.imread(image_dir + file)

    pipeline.process(img)

    hulls = pipeline.filter_contours_output


    for i in range(len(hulls)):
        cv2.drawContours(img, hulls, i, (0, 0, 255), 1)
    
    pts = []
    for cnt in hulls:
        rect = cv2.minAreaRect(cnt) # TODO find a better way to extract points
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(0,0,255),2)
        
        box = box[np.lexsort((box[:,0],box[:,1]))]

        # ugly way to find the top two points
        cv2.circle(img, (box[0]), 4, (170, 255, 255), -1)
        other = box[1]
        if(abs(box[0][0] - other[0]) < 10):
            other = box[2]
        cv2.circle(img, (other), 4, (170, 255, 255), -1)

        pts.append(box[0])
        pts.append(other)

    if(len(pts) < 5):
        continue

    pts = np.array(pts)
    ellipse = cv2.fitEllipse(pts)
    cv2.ellipse(img,ellipse,(0,255,0),2)

    
    show_image(img)
