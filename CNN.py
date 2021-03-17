
"""
Created on Mon 

@author: asellent
"""

import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
from basics import *
def coco(image):
#%% constants
# load YOLO object detector trained on COCO dataset (80 classes)
    weightsPath = './coco/frozen_inference_graph.pb'
    configPath = './coco/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt'

    classesNamesPath = './coco/coco.names'

#test settings
    confThreshold = 0.5
    nmsThres = 0.3
    imagePath = './coco/DSCN6255.JPG'


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
    #image = scaleImage(image)
    #cv.imshow("COCO", image)

    #cv.waitKey(0)
    #cv.destroyAllWindows()     
    return(image)        

def yolo(image):
    #%% constants
# load YOLO object detector trained on COCO dataset (80 classes)
    weightsPath = './yolo/yolov3.weights'
    configPath = './yolo/yolov3.cfg'

    classesNamesPath = './coco/coco.names'

#test settings
    confThres = 0.5
    nmsThres = 0.3
    #imagePath = 'data/000000579635.jpg'

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
    height, width = image.shape[:2]

#%% prepare input and feed ot through the net
    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)

#ln = net.getUnconnectedOutLayersNames()
#layerOutputs = net.forward(ln)
    output = net.forward()

#%% process the returned data

    boxes = []
    probs = []
    classIDs = []

# loop over each of the detections
    for detection in output:

        scores = detection[5:]
        classID = np.argmax(scores)
        probability = scores[classID]
    # filter out weak predictions by ensuring the detected
    # probability is greater than the minimum probability
        if probability > confThres:
        # scale the bounding box coordinates back relative to the
        # size of the image, keeping in mind that YOLO actually
        # returns the center (x, y)-coordinates of the bounding
        # box followed by the boxes' width and height
            box = detection[0:4] * np.array([width, height, width, height])
            (centerX, centerY, w, h) = box.astype("int")
        # use the center (x, y)-coordinates to derive the top and
        # and left corner of the bounding box
            x = int(centerX - (w / 2))
            y = int(centerY - (h / 2))
        # update our list of bounding box coordinates, confidences,
        # and class IDs
            boxes.append([x, y, int(w), int(h)])
            probs.append(float(probability))
            classIDs.append(classID)
            
# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
    idxs = cv.dnn.NMSBoxes(boxes, probs, score_threshold=confThres, nms_threshold=nmsThres)

# ensure at least one detection exists
    if len(idxs) > 0:
	# loop over the indexes we are keeping
    	for i in idxs.flatten():
		# extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
              #a bounding box rectangle and label on the image
            color = (255, 255, 255)
            cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], probs[i])
            cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
               # show the output image
   
    #cv.namedWindow('YOLO', cv2.WINDOW_NORMAL)
    #scale = 1000 / np.max(image.shape)
    #x, y = image.shape[0]*scale, image.shape[1]*scale
    #cv.resizeWindow('YOLO', 800, 600)
    #image = scaleImage(image)
    #cv.imshow("YOLO", image)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    return(image)             
