# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:08:34 2021

@author: Simon
"""

#####################################################################################################################################################
#
# flags
#
##################################################################################################################################################### 

#size for downscaling
IMG_TARGET_SIZE = 1000
#thresholds for modifiy threshold
THRESHOLD_MIN = 20
THRESHOLD_MAX = 255
#threshold for hough based crop
CUT_THRESH = 150
#change input path (default = same path as pyhon files)
USE_ABSOLUTE_PATH = True
ABSOLUTE_PATH = "C:\\Users\\Simon\\Desktop\\masterarbeit\\contours"
#use iterative instead of adaptive threshold
MODIFY_THRESHOLD = False
#search for a template to get aspect ratio
USE_TEMPLATE = True
#use simple crop instead of the more complex hough based crop
SIMPLE_CROP = False
#use cut based on contours instead of rects
CONT_BASED_CUT = True
#use CNN: yolo, coco, both
USE_CNN = ""
#evaluate OCR
EVALUATE = True
#only use picture with this name(s)
CHECK_PICTURE = "1059"


################################################################
COUNTER = 0