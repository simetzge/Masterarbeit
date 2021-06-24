# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:08:34 2021

@author: Simon Metzger

licensend under Attribution-NonCommercial-ShareAlike 3.0 Germany

CC BY-NC-SA 3.0 DE
"""

#####################################################################################################################################################
#
# flags
#
##################################################################################################################################################### 

#change input path (default = same path as pyhon files)
USE_ABSOLUTE_PATH = True
ABSOLUTE_PATH = "C:\\Users\\Simon\\Desktop\\masterarbeit\\contours"
#input data format
INPUT_FORMAT = ".jpg"

#only use picture with this name
CHECK_PICTURE = "test"


#use OCR
OCR = True
#use CNN: yolo, coco, both
USE_CNN = ""


#####################################################################################################################################################
#
# rect detection
#
##################################################################################################################################################### 

#use iterative instead of adaptive threshold
MODIFY_THRESHOLD = False
#search for a template to get aspect ratio
USE_TEMPLATE = True
#use simple crop instead of the more complex hough based crop
SIMPLE_CROP = True
#use cut based on contours instead of rects (not recommended)
CONT_BASED_CUT = False

#size for downscaling
IMG_TARGET_SIZE = 1000
#min size of rectangle
MIN_RECT_AREA = 2000
#thresholds for modifiy threshold
THRESHOLD_MIN = 20
THRESHOLD_MAX = 200
#threshold for hough based crop
CUT_THRESH = 120

#####################################################################################################################################################
#
# OCR
#
##################################################################################################################################################### 

#tesseract path
TESS_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#tesseract config
OCR_CONFIG = ("board -l dict --oem 1 --psm 3")

#invert image (use this if the background is darker than the text)
INVERT_IMAGE = True

#####################################################################################################################################################
#
# evaluation
#
##################################################################################################################################################### 

#evaluate OCR
EVALUATE = False
#name of evaluation list
EVALUATION_LIST = "evaluationlist.csv"
#add optmimum value to evaluation
OPTIMUM = True
#F-Score instead of ratio
FSCORE = False
#add recall, precision to evaluation
ALL_MEASURES = False
