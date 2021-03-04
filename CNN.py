
"""
Created on Mon 

@author: asellent
"""

import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
from basics import *
def CNN(image):
#%% constants
# load YOLO object detector trained on COCO dataset (80 classes)
    weightsPath = './10/frozen_inference_graph.pb'
    configPath = './10/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt'

    classesNamesPath = './10/coco.names'

#test settings
    confThreshold = 0.5
    nmsThres = 0.3
    imagePath = './10/DSCN6255.JPG'


#%% load data
#CNN
    net = cv.dnn.readNet(configPath, weightsPath)

#class names
    classes = []
    with open(classesNamesPath) as f:
        for line in f:
            classes.append(line.rstrip('\n'))

#image
    #image = cv.imread(imagePath)

    frameHeight, frameWidth = image.shape[:2]

#%% prepare input and feed ot through the net
    blob = cv.dnn.blobFromImage(image[:,:,:],swapRB=True, crop=False)
    net.setInput(blob)

# #ln = net.getUnconnectedOutLayersNames()
# #layerOutputs = net.forward(ln)
    output = net.forward()
# Network produces output blob with a shape 1x1xNx7 where N is a number of
# detections and an every detection is a vector of values
# [batchId, classId, confidence, left, top, right, bottom]

    boxes = []
    probs = []
    classIds = []

    out = output[0]#no batches
    for detection in out[0]:
        confidence = detection[2]
        if confidence > confThreshold:#optional
            left = int(detection[3] * frameWidth)
            top = int(detection[4] * frameHeight)
            right = int(detection[5] * frameWidth)
            bottom = int(detection[6] * frameHeight)
            width = right - left + 1#OpenCV: last point not included
            height = bottom - top + 1
            
            classIds.append(int(detection[1]))  # Skip background label
            probs.append(float(confidence))
            boxes.append([left, top, width, height])        

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
    idxs = cv.dnn.NMSBoxes(boxes, probs, score_threshold=confThreshold, nms_threshold=nmsThres)
            
# ensure at least one detection exists
    if len(idxs) > 0:
	# loop over the indexes we are keeping
    	for i in idxs.flatten():
            if (i > len(boxes)) or (i > len(probs)) or (i > len(classIds)) or (classIds[i] > len(classes)):
                break
		# extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
		# draw a bounding box rectangle and label on the image
            color = (0, 255, 0)
            cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIds[i]], probs[i])
            cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print(text)
# show the output image
    cv.imshow("Image", image)

    cv.waitKey(0)
    cv.destroyAllWindows()             
