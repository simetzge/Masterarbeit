# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:39:38 2020

@author: Simon
"""

import numpy as np
import cv2
import os
import re
import math
from ocr import *
from CNN import *

#####################################################################################################################################################
#
# flags
#
##################################################################################################################################################### 

IMG_TARGET_SIZE = 1000
THRESHOLD_MIN = 20
THRESHOLD_MAX = 255
MODIFY_THRESHOLD = False
USE_TEMPLATE = True
USE_ABSOLUTE_PATH = True
SIMPLE_CROP = False
CUT_THRESH = 150
ABSOLUTE_PATH = "C:\\Users\\Simon\\Desktop\\masterarbeit\\contours"
CHECK_PICTURE = ""
################################################################
COUNTER = 0

try:   
#####################################################################################################################################################
    
    def main():
        
        ##########################
        #a few tests with files
        #if TESTFLAG == True:
         #   print ('works')
        #testocr()
        ###########################
        
        filePaths, fileNames = searchFiles('.jpg')
    
        images = []
    
        #images = [cv2.imread(files, cv2.IMREAD_GRAYSCALE) for files in filePaths]
        images = [cv2.imread(files) for files in filePaths]
    
        images = [scaleImage(img) for img in images]
        
        #get template from aspect ratio if flag is set
        if USE_TEMPLATE == True:
            
            getAspectRatio(images, fileNames)
            
        #detect rectangles in every image, adaptive or iterative
        for i, img  in enumerate(images):
            #skip template
            if 'template' in fileNames[i]:
                continue
            #skip all pictures but the one that should be checked
            if CHECK_PICTURE != "":
                if not CHECK_PICTURE in fileNames[i]:
                    continue      
            print("the next image is " + fileNames[i])
            if MODIFY_THRESHOLD:
                rects = rect_detect_iterative(img, fileNames[i])
            else:
                rects = rect_detect_adaptive(img, fileNames[i])
            cut(img, rects, fileNames[i])
            #CNN(img)
        print(COUNTER)


#####################################################################################################################################################
#    
# function for searching all files with the matching extension in the input directory
# will return the paths and names of all files found
#
#####################################################################################################################################################

    def searchFiles(extension):
        
        #get skript path
        if USE_ABSOLUTE_PATH == True:
            path = ABSOLUTE_PATH
        else:
            path = os.getcwd()
        
        #list all files in path
        dirs = os.listdir(path)
        
        #mnake empty array for input files
        files = []
        names = []
        
        #if input dir found
        if 'input' in dirs:
            print('input gefunden')
            
            #list all files in input dir
            content = os.listdir(path + '\\input')
            
            #match the files with given extension
            for item in content:
                jregex = re.compile(extension, re.IGNORECASE)
                match = jregex.search(item)
                #if found add to file array
                if match != None:
                    files.append(path + '\\input\\' + item)
                    names.append(item)
        #print note and end skript if no input dir
        else:
            print('input fehlt')
            exit()
        #return all found files
        return(files, names)

#####################################################################################################################################################
#
# writes given images with given names and an extension for the modification, e.g. 'blurred'
# will create outputfolder in program path
#
#####################################################################################################################################################

    def output(folder, img, name, mod=''):
        
        if USE_ABSOLUTE_PATH == True:
            path = ABSOLUTE_PATH
        else:
            path = os.getcwd()
        #list all files in path
        dirs = os.listdir(path)
        if folder in dirs:
            print(folder + '-Ordner vorhanden')
        else:
            os.makedirs(path + '\\' + folder)
        if len(mod) == 0:
            cv2.imwrite(path + '\\' + folder + '\\' + name[:-4] + '.png', img)
        else:
            cv2.imwrite(path + '\\' + folder + '\\' + name[:-4] + '_' + mod + '.png', img)
        
#####################################################################################################################################################
#      
# resize image to max 1000p
#
#####################################################################################################################################################

    def scaleImage(img):
        scale = IMG_TARGET_SIZE / np.max(img.shape)
        return (cv2.resize(img, (0,0), fx = scale, fy = scale))

#####################################################################################################################################################
#      
# normalize image to range from 0 to 255
#
#####################################################################################################################################################
    
    def normalizeImage(img):
        
        #img = scaleImage(img)
        (x, y) = img.shape
        normImg = np.zeros((x,y))
        img = cv2.normalize(img,  normImg, 0, 255, cv2.NORM_MINMAX)
        return (img)
        
    
##################################################################################################################################################### 
#
# sets an adaptive threshold, sends the results to rect_detect and those results to output
#
#####################################################################################################################################################

    def rect_detect_adaptive(img, fileName):
        
        #convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        gray = normalizeImage(gray)
        
        binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,1)

        #findcontours
        contours, rois = rect_detect(binary) 
        
        #print the number of rectangles for debug reasons
        #print(len(rois))    
        
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        #add contours in red to image
        roisImg = cv2.drawContours(gray, contours, -1, (0, 0, 230))
        
        #add the found rectangles in green to image
        roisImg = cv2.drawContours(roisImg, [cv2.boxPoints(rect).astype('int32') for rect in rois], -1, (0, 230, 0))
        
        #send the modified images in the output function
        output('output', roisImg, fileName, 'adaptive')
        
        #cut(img, rois, fileName)
        return(rois)
    
#####################################################################################################################################################
#
# sets an increasing threshold, sends the results to rect_detect and those results to output
#
#####################################################################################################################################################

    def rect_detect_iterative(img, fileName):
        
        j = THRESHOLD_MIN
        
        allRois = []
        
        #convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        gray = normalizeImage(gray)
        
        while j <= 200:
            
            rois = []
            
            contours = []
            
            ret, binary = cv2.threshold(gray, j, THRESHOLD_MAX, cv2.THRESH_BINARY)
            
            contours, rois = rect_detect(binary)
        
            #print the number of rectangles for debug reasons
            #print(len(rois))    
        
            #add contours in red to image
            #roisImg = cv2.drawContours(img, contours, -1, (0, 0, 230))
        
            #add the found rectangles in green to image
            #roisImg = cv2.drawContours(roisImg, [cv2.boxPoints(rect).astype('int32') for rect in rois], -1, (0, 0, 250))
        
            #send the modified images in the output function
            #output(roisImg, fileName, str(j))

            if len(rois) > 0:
                
                allRois.append(rois)
            
            j += 5
        
        #new rois list
        rois_list = []
        
        #go through the found rectangles and add them to an array of dictionaries
        for r in allRois:
            
            for i in range(len(r)):
                (x,y), (w,h), angle = r[i]
                rois_dict = {
                    }
                rois_dict["x"] = x
                rois_dict["y"] = y
                rois_dict["w"] = w
                rois_dict["h"] = h
                rois_dict["angle"] = angle
                rois_dict["same"] = 0
                
                rois_list.append(rois_dict)
        
        #find and count rectangles in the same area
        for i in range (len(rois_list)):
            for j in range (len(rois_list)):
                recta = rois_list[i]["x"], rois_list[i]["y"],rois_list[i]["w"],rois_list[i]["h"]
                rectb = rois_list[j]["x"], rois_list[j]["y"],rois_list[j]["w"],rois_list[j]["h"]
                if intersection_over_union(recta, rectb) > 0.9:
                    rois_list[i]["same"] = rois_list[i]["same"] + 1
                    rois_list[j]["same"] = rois_list[j]["same"] + 1
        #new rectangle list
        rects = []
        #same is the number of same rois in the area, 0 is default
        same = 0
        
        #if there are dictionaries in the list, search for the one with the highest number of same rectangles in the area
        if len(rois_list) > 0:
            roi = rois_list[0]
            for  i in range (len(rois_list)):
                if rois_list[i]["same"] >= roi["same"]:
                    roi = rois_list[i]
            #print number of same rois for debug reasons
            #print(roi["same"]) 
        #add contours in red to image
            if roi["same"] >= 6:
                
                #roisImg = cv2.drawContours(gray, contours, -1, (0, 0, 230))
                rect = (roi["x"],roi["y"]),(roi["w"],roi["h"]),roi["angle"]
                rects.append(rect)
                same = roi["same"]
        
            #convert to grayscale
            #gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
            #add the found rectangles in green to image
            #roisImg = cv2.drawContours(gray, [cv2.boxPoints(rect).astype('int32') for rect in rects], -1, (0, 230, 0))
        
        #convert to grayscale
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        #add the found rectangles in green to image
        roisImg = cv2.drawContours(gray, [cv2.boxPoints(rect).astype('int32') for rect in rects], -1, (0, 230, 0))
                    
        #send the modified images in the output function
        output('output', roisImg, fileName, str(same))
        #cut(img, rects, fileName)
        return(rects)
        
#####################################################################################################################################################
#
# detetects all rectangles in a given binary image
#
#####################################################################################################################################################
        
    def rect_detect(binary):
        
        #findcontours
        contours, hierarchy  = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        #creat array for regions of interest
        rois = []
        
        #move through every contour in array contours
        for contour in contours:
            
            #compute contour area
            contArea = cv2.contourArea(contour)
            
            #throw out too small areas
            if not 1000 < contArea:
                continue            
            
            #create rectangle around contours
            (x, y), (w, h), angle = rect = cv2.minAreaRect(contour)
            
            #compute area of this rectangle
            rectArea = w * h
            
            #compare the areas to each other, make sure they don't differ too much
            if contArea / rectArea < 0.85:
                continue
            
            #ignore contours as big as the image
            if w > binary.shape[0] * 0.9 or h > binary.shape[1]*0.9:
                continue
            
            #compute if area is not empty
            if rectArea != 0:
               
                #if template is used, check for aspect ratio
                if USE_TEMPLATE == True and aspectRatio != 0:
                    
                    #get aspect ratios of rect and approx
                    asra = max(w,h)/min(w,h)
                    
                    #ignore this shape if aspect ratio doesn't fit
                    if not (asra < aspectRatio * 1.3 and asra > aspectRatio *0.7):
                        continue
                    #print aspect ratio for debug reasons
                    #print ("asra " + str(asra))                    
                
                #else aspect ratio should be max 2:1
                else:
                    #make sure the aspect ratio is max 2:1
                    if max(w,h) > 2 * min(w,h):
                        continue
                
            #if every condition is met, save the rectangle area in the array
            rois.append(rect)
        
        return (contours, rois)
    
#####################################################################################################################################################
#
# gets global aspect ratio from template
#
#####################################################################################################################################################
    
    def getAspectRatio(imgs, fnames):
        
        #define global aspectRatio, default is 0 in case no file is found
        global aspectRatio
        aspectRatio = 0
        img = None
        
        #search for a file named template
        for i, names in enumerate(fnames):
            jregex = re.compile('template', re.IGNORECASE)
            match = jregex.search(names)
            
            #break if match is found
            if match != None:
                img = imgs[i]
                print('template found')
                break
        #print error if no match in the entire input is found
        if match == None:
            print("failed to find template")
        else:
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,1)

            #detect template
            contours, rois = rect_detect(binary) 
            
            #compute aspect ratio, write in global variable
            (x, y), (w, h), angle = rois[0]            
            aspectRatio = round(max(w, h)/ min (w,h), 3)

#####################################################################################################################################################
#
# classic intersection over union
# returns 0 (no intersection) to 1 (perfect intersection))
#
#####################################################################################################################################################

    def intersection_over_union(recta, rectb):
        
        #top left coordinates
        xa = max(recta[0], rectb[0])
        ya = max(recta[1], rectb[1])
        #bottom right coordinates
        xb = min(recta[0] + recta[2], rectb[0] + rectb[2])
        yb = min(recta[1] + recta[3], recta[1] + rectb[3])
        
        intersection = max(0, xb - xa) * max(0, yb - ya)
        
        rectareaa = recta[2] * recta[3]
        rectareab = rectb[2] * rectb[3]
        
        iou = round(intersection / float(rectareaa + rectareab - intersection), 3)
        
        #print(iou)
        
        return(iou)

#####################################################################################################################################################
#
# cuts rectangle from image
# sends both, modified image and rectangle, to output
#
#####################################################################################################################################################

    def cut(img, rects, fileName):
        
        # generate mask for extraction of rectangles
        mask = np.zeros(img.shape[:2], dtype=bool)
        
        # crop every rectangle with simple crop ()
        for i, rect in enumerate(rects):
            
            (x, y), (w, h), angle = rect
        
            bl, br, tr , tl = cv2.boxPoints(rect).astype('int32')
            
            #crop = img[min(tl[1],br[1]): max(tl[1],br[1]),min(tl[0],br[0]):max(tl[0],br[0])]
            
            if SIMPLE_CROP:
                #old version, works, but not perfect
                crop = rotate_board (img, rect)
            else:
                # new version
                crop = hough_rotate(img,rect, CUT_THRESH)
                
            #output('rectanglecut', rectcut, fileName)
            
            #end function if no crop image found (hough rotate returns [None] if something went wrong)
            if len(crop) < 2:
                print(fileName + " failed")
                continue

            #crop = new_preprocessing (crop)
            crop = new_preprocessing (crop)
            
            #ocr
            #################################
            text, rotate = image_to_text(crop)
            
            if rotate == True:
                crop = cv2.rotate(crop, cv2.cv2.ROTATE_180)
            
            
            #write text on image
            cv2.putText(crop, text, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            output('rect', crop, fileName, str(i)) 
            
            # mask area with size of detected rect
            #mask[min(tl[1],br[1]): max(tl[1],br[1]),min(tl[0],br[0]):max(tl[0],br[0])] = True
            
            # mask area sligtly bigger than detected rect to cut the complete board with its border
            mask[int(min(tl[1],br[1]) - 0.1 * w): int(max(tl[1],br[1]) + 0.1 * w),int(min(tl[0],br[0]) - 0.1 * h):int(max(tl[0],br[0]) + 0.1 * h)] = True
        
        #modify image: set mask area to black
        imgcut = img.copy()
        rectcut = imgcut[mask]
        imgcut[mask] = 0

        #send the modified images in the output function
        output('imagecut', imgcut, fileName)
    

#####################################################################################################################################################
#
# create binary images for ocr
#
#####################################################################################################################################################

    def preprocessing_old(img):
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # multiple blurring and normalization to get better contours
        for i in range (100):
            
            median = cv2.medianBlur(gray, 3)
        
            gray = normalizeImage(median)
            
            # set everything lower than 60 to 0
            gray = np.where(gray < 60, 0, gray)
            
            if i % 10 == 0:
                
                gray = cv2.fastNlMeansDenoising(gray,7,7,7)
                
                #try to set push values in different directiosn
                #gray = np.where(gray < 130, 105, gray)
                #gray = np.where(gray < 100, 80, gray)
                #gray = np.where(gray < 75, 51, gray)
                #gray = np.where(gray < 50, 0, gray)
                #gray = np.where(gray > 160, 175, gray)
                
                # set everything lower than 60 to 0
                #gray = np.where(gray < 60, 0, gray)
                
                # change low and high based on mean
                #gray = np.where(gray < np.mean(gray), 0, gray)
                #gray = np.where(gray > ((np.mean(gray)+255)/2), 255, gray)
                
                # change low based on mean
                #gray = np.where(gray < np.mean(gray), 0, gray)
                
                #change values based on mean without zeros (black area)
                #gray = np.where(gray < np.mean(gray), 0, gray)
                #newgray = gray.copy()
                #newgray = newgray[newgray!=0]
                #mean = np.mean(newgray)
                #print (mean, np.mean(gray))
                #gray = np.where(gray > mean, 250, gray)
                
                #gray [gray == nan] = 0

                # change high based on mean
                #gray = np.where(gray > np.mean(gray), 150, gray)
                 
                #ret, gray = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY,cv2.THRESH_OTSU)
        
                #gray = cv2.GaussianBlur(gray,(1,1),0)
                
                
                #gray = skeleton(gray)
        #newgray = gray.copy()
        #newgray = newgray[newgray!=0]
        #mean = np.mean(newgray)
        #print (mean, np.mean(gray))
        #gray = np.where(gray > mean, 250, gray)
        #gray = np.where(gray < np.mean(gray), 0, gray)        
        #gray = np.where(gray > np.mean(gray), 200, gray)      
        #gray = cv2.equalizeHist(gray)  
        #gray = cv2.fastNlMeansDenoising(gray,7,7,7)
        
        #ret, gray = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        
        #kernel = np.ones((5,5),np.uint8)
        
        #dilate = cv2.dilate(gray , kernel, iterations = 1)
        
        #gray = skeleton(gray)
        
        #median = cv2.GaussianBlur(median,(3,3),5)        
        #gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        #ret, img = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY,cv2.THRESH_OTSU) #imgf contains Binary image
        #img = scaleImage(img)
        #gray = cv2.adaptiveThreshold(gray.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
        
        #kernel = np.ones((1,1),np.uint8)
        
        #gray = cv2.erode(gray,kernel,iterations = 10)
        
        #openening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        #closing = cv2.morphologyEx(openening, cv2.MORPH_CLOSE, kernel)
        
        #gray = closing
        
        #ret, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
        #img = cv2.GaussianBlur(img,(1,1),0)
        #ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        #img = cv2.GaussianBlur(img,(1,1),0)
        
        #ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY  + cv2.THRESH_OTSU)
        #img = cv2.GaussianBlur(img,(1,1),0)
        #img = cv2.bitwise_or(img, closing)
        #img = cv2.GaussianBlur(img,(1,1),0)

               
        #img = normalizeImage(img)
        #img = cv2.GaussianBlur(img,(1,1),0)
        #ret, img = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY)
        
        #img = skeleton(img)
        
        #ret, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
        
        #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,1)
        #img = cv2.GaussianBlur(img,(5,5),0)
        
        #ret, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
        
        #############################################
        # maybe try se canny edge ja?
        #############################################        
        
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #gray = cv2.medianBlur(gray,5)
        
        #gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        #kernel = np.ones((5,5),np.uint8)
        
        #gray = cv2.dilate(gray, kernel, iterations = 1)
        
        #gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
                
        return(gray)
    
    #cannyedge from opencv doc
    def cannyThreshold(img):
        max_lowThreshold = 100
        ratio = 3
        kernel_size = 3
        low_threshold = 1
        img_blur = cv2.blur(img, (3,3))
        detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
        mask = detected_edges != 0
        dst = img * (mask[:,:].astype(img.dtype))
        return (dst)
    
    #skeleton from opencv doc
    def skeleton(img):
        # Step 1: Create an empty skeleton
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)
        
        # Get a Cross Shaped Kernel
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        
        # Repeat steps 2-4
        while True:
            #Step 2: Open the image
            open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            #Step 3: Substract open from the original image
            temp = cv2.subtract(img, open)
            #Step 4: Erode the original image and refine the skeleton
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel,temp)
            img = eroded.copy()
            # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
            if cv2.countNonZero(img)==0:
                break
        return(skel)
    
#####################################################################################################################################################
#
# crop rotated rectangle with warpperspective
#
#####################################################################################################################################################

    def rotate_board(img, rect):
        
        #get boxpoints
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        (x, y), (w, h), angle = rect
        #cast boxpoints for source
        src = box.astype("float32")
        #get array for destination
        dst = np.array([[0, h],[0, 0],[w, 0],[w, h]], dtype="float32")
        
        #get rotation matrix
        M = cv2.getPerspectiveTransform(src, dst)
        
        #warp
        warped = cv2.warpPerspective(img, M, (int(w), int(h)))
        
        if warped.shape[0] > warped.shape[1]:
            #warped = np.rot90(warped)
            warped = cv2.rotate(warped, cv2.cv2.ROTATE_90_CLOCKWISE)
        
        # dsize
        if USE_TEMPLATE == True:
            dsize = (IMG_TARGET_SIZE, int(IMG_TARGET_SIZE / aspectRatio))
        else:
            dsize = (IMG_TARGET_SIZE, int(IMG_TARGET_SIZE * 0.8))

        # resize image
        warped = cv2.resize(warped, dsize, interpolation = cv2.INTER_AREA)
        
        return (warped)

#####################################################################################################################################################
#
# better crop with Hough-Transform
#
#####################################################################################################################################################

    def hough_rotate(img, rect, threshold):
        
        debug_hough = False
        
        #if threshold is too low, use simple crop
        if threshold < 100:
            return (rotate_board(img, rect))
            
        #how much bigger the crop image is than the board
        #sizeFactor = 0.25
        (x, y), (w, h), angle = rect
        #sizeFactor should increase with angle size, minimum should be 0.2
        sizeFactor = max(round(0.003 * max(angle, -angle),1), 0.2)
        print (sizeFactor)
        #crop image with a larger area than the detected rect to get the corners of the board
        bl, br, tr , tl = cv2.boxPoints(rect).astype('int32')
        #left = int(max(min(tl[0],br[0]) - sizeFactor * w,1))
        #right = int(min(max(tl[0],br[0]) + sizeFactor * w, w))
        #top = int(max(min(tl[1],br[1]) - sizeFactor * h,1))
        #bottom = int(min(max(tl[1],br[1]) + sizeFactor * h,h))
        
        left = min(tl[0],br[0]) - sizeFactor * h
        right = max(tl[0],br[0]) + sizeFactor * h
        top = min(tl[1],br[1]) - sizeFactor * w
        bottom = max(tl[1],br[1]) + sizeFactor * w
        if left < 0: left = 0
        if right > img.shape[1]: right = img.shape[1]
        if top < 0: top = 0
        if bottom > img.shape[0]: bottom = img.shape[0]
        left = int(left)
        right = int(right)
        top = int(top)
        bottom = int(bottom)
        
        crop_img = img[top:bottom,left:right]
        #crop_img = img[int(min(tl[1],br[1]) - sizeFactor * w): int(max(tl[1],br[1]) + sizeFactor * w),int(min(tl[0],br[0]) - sizeFactor * h):int(max(tl[0],br[0]) + sizeFactor * h)]
        
        new_rect = (x,y), (int(w*1.3), int(h*1.3)), angle
        crop_img = rotate_board(img, new_rect)
        
        if crop_img.shape[0] > crop_img.shape[1]:
            #warped = np.rot90(warped)
            crop_img = cv2.rotate(crop_img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        #preprocessing: scale, blur, grayscale, normalize, binary threshold 180, blur, skeleton, blur
        crop_img = scaleImage(crop_img)
        blur = cv2.bilateralFilter(crop_img,9,75,75)
        blur = cv2.fastNlMeansDenoising(blur,7,7,15)        
        blur = cv2.GaussianBlur(blur,(7,7),15)
        
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)       
        gray = normalizeImage(gray)        
        mean = np.mean(gray)
        #mean *1.2 bisher zweitbeste (31), beste mean+25 (27)
        
        #gray = preprocessing(crop_img)
        
        ret, binary = cv2.threshold(gray, int(mean+30), THRESHOLD_MAX, cv2.THRESH_BINARY)
        #binary = cv2.adaptiveThreshold(binary,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,1)
        
        if debug_hough:
            cv2.imshow("test", binary)
            cv2.waitKey()
        binary = cv2.GaussianBlur(binary,(3,3),15)           
        binary = skeleton (binary)
        binary = cv2.GaussianBlur(binary,(3,3),15)
        
        #get shape
        height, width = binary.shape 
        
        # cannyedge        
        dst = cannyThreshold(binary)
        #hough with canny edge
        lines = cv2.HoughLines(dst, 1, np.pi / 180, threshold, None, 0, 0)
        
        cdst = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # empty lineList to collect all lines        
        lineList = []
        interList = []        
        
        if lines is not None:            
            # go through lines, calculate the coordinates
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(cdst, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
                # add lines to List
                line =[pt1,pt2]
                lineList.append(line)                
                
            # calculate every intersection between lines 
            for i in range(0, len(lineList)):    
                for j in range(0, len(lineList)):
                    # skip intersection of line with itself
                    if lineList[i] == lineList[j]:
                        continue
                    
                    #skip if lines are in the same direction
                    quia = getQuadrant(binary, lineList[i][0])
                    quib = getQuadrant(binary, lineList[i][1])
                    quja = getQuadrant(binary, lineList[j][0])
                    qujb = getQuadrant(binary, lineList[j][1]) 
                    #if (quia == quja or quia == qujb) and (quib == quja or quib == qujb):
                    if (quia == quja or quia == qujb) and (quib == quja or quib == qujb):
                        continue
                    
                    # call intersection calculation
                    inter = intersection(lineList[i], lineList[j])
                    #ignor if the intersection is on the corners
                    
                    if not ((inter[0] < 1 or inter [0] > width) or inter[1] < 1 or inter[1] > height):
                        
                        interList.append(inter)
                        # add intersections as dots to output image for visualization
                        cdst = cv2.circle(cdst, inter, 4, (0,255,0), 4)
                    else:
                        cdst = cv2.circle(cdst, inter, 4, (0,0,255), 4)
            if debug_hough:       
                cv2.imshow("test", cdst)
                cv2.waitKey()
        
        tlList = []
        trList = []
        blList = []
        brList = []
        
        #sort inter
        for inters in interList:
            if getQuadrant(binary, inters) == "tl":
                tlList.append(inters)

            if getQuadrant(binary, inters) == "tr":
                trList.append(inters)
                
            if getQuadrant(binary, inters) == "bl":
                blList.append(inters)
                
            if getQuadrant(binary, inters) == "br":
                brList.append(inters)
        
        #cast tuple to list
        tl = getCorner(tlList)   
        tr = getCorner(trList)
        bl = getCorner(blList)
        br = getCorner(brList)
        
        #when no corner detected return simple cropped image
        if tl == None or tr == None or bl == None or br == None:
            global COUNTER
            COUNTER = COUNTER +1
            #return (rotate_board(img, rect))
            #perform hough rotate with lower threshold
            return(hough_rotate(img, rect, threshold-5))
            
        
        tl = list(tl)     
        tr = list(tr)
        bl = list(bl)
        br = list(br)
        
        #put points in array
        src = [bl, tl, tr, br]
        #get array for destination
        dst = np.array([[0, height],[0, 0],[width, 0],[width, height]], dtype="float32")
        print (src)
        #get rotation matrix
        M = cv2.getPerspectiveTransform(np.float32(src), dst) 
        #warp
        warped = cv2.warpPerspective(crop_img, M, (int(width), int(height)))
        
        # dsize
        if USE_TEMPLATE == True:
            dsize = (warped.shape[1], int(warped.shape[1] / aspectRatio))
        else:
            dsize = (warped.shape[1], int(warped.shape[1] * 0.8))

        # resize image
        warped = cv2.resize(warped, dsize, interpolation = cv2.INTER_AREA)
        
        # visualization for debug
        cdst = crop_img
        cdst = cv2.drawContours(cdst, [cv2.boxPoints(((tl[0], tl[1]), (10, 10), 0)).astype('int32')], -1, (250, 0, 250))
        cdst = cv2.drawContours(cdst, [cv2.boxPoints(((tr[0], tr[1]), (10, 10), 0)).astype('int32')], -1, (250, 0, 250))
        cdst = cv2.drawContours(cdst, [cv2.boxPoints(((bl[0], bl[1]), (10, 10), 0)).astype('int32')], -1, (250, 0, 250))
        cdst = cv2.drawContours(cdst, [cv2.boxPoints(((br[0], br[1]), (10, 10), 0)).astype('int32')], -1, (250, 0, 250))   
        if debug_hough:
            cv2.imshow("test", cdst)
            cv2.waitKey()
        return (warped)
        
#####################################################################################################################################################
#
# calculate the coordinate with the most intersections arround
#
#####################################################################################################################################################

    def getCorner(inList):
        
        # when list empty return 0
        if len(inList) == 0:
            return(None)
        #empty intersection over union list
        iouList = []
        #compare every item in list with every other item in list
        for i in range(len(inList)):
            iou = 0
            for j in range(len(inList)):
                if inList[i] != inList[j]:
                    #build rectangles around coordinates and check via intersection over union if they are close to each other
                    recta = inList[i][0], inList[i][1],10,10
                    rectb = inList[j][0], inList[j][1],10,10
                    if intersection_over_union(recta, rectb) > 0.8:
                        #if close, counter +1
                        iou += 1
            #save the counters in list
            iouList.append(iou)
        #sort the counter list, return the coordinates with the highest counter
        position = np.argsort(iouList)
        corner = inList[position[-1]]
        return(corner)
    
#####################################################################################################################################################
#
# return the abbreviation of the quadrant where the point is located
#
#####################################################################################################################################################

    def getQuadrant(img, coordinate):
        
        height, width = img.shape
        if coordinate[0] < width / 2 and coordinate[1] < height / 2:
            return("tl")
                
        if coordinate[0] > width / 2 and coordinate[1] < height / 2:
            return("tr")
                
        if coordinate[0] < width / 2 and coordinate[1] > height / 2:
            return("bl")
                
        if coordinate[0] > width / 2 and coordinate[1] > height / 2:
            return("br")
     
#####################################################################################################################################################
#
# compute intersection of two lines
# cast to int for use in images
#
#####################################################################################################################################################

    def intersection(lineA, lineB):
        
        xdiff = (lineA[0][0] - lineA[1][0], lineB[0][0] - lineB[1][0])
        ydiff = (lineA[0][1] - lineA[1][1], lineB[0][1] - lineB[1][1])
        
        div = det(xdiff, ydiff)
        x = 0
        y = 0
        if div:           
            d = (det (*lineA), det(*lineB))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
        return int(x),int(y)
        
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    
#####################################################################################################################################################
#
# warp the board back to its former location to get a better fitting mask, failed and not used atm
#
#####################################################################################################################################################
    
    def get_mask(img, rect, board):
        
        #get boxpoints
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        (x, y), (w, h), angle = rect
        #cast boxpoints for source
        dst = box.astype("float32")
        #get array for destination
        src = board
        #get rotation matrix
        M = cv2.getPerspectiveTransform(src, dst)
        #warp
        warped = cv2.warpPerspective(img, M, (int(w), int(h)))       
        
        return(warped)
    
#####################################################################################################################################################
#
# rotate the whole image to get a better fitting mask, failed and not used atm
#
#####################################################################################################################################################

    def rotate_image(img, rect, mask):
        
        #creat mask
        new_mask = np.zeros(img.shape[:2], dtype=bool)
        (x, y), (w, h), angle = rect
        #get height and width from image
        imgh, imgw = img.shape[0], img.shape[1]
        #get roation matrix
        M = cv2.getRotationMatrix2D((x,y), angle, 1)
        #rotate image
        img_rot = cv2.warpAffine(img, M, (imgw, imgh))
        
        #cast to int, so the coordinates can be used as indices
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        #set mask to true in the area of the rectangle
        new_mask[y-int(h/2):y+int(h/2),x-int(w/2):x+int(w/2)] = True
        
        #cut the rectangle out with mask
        img_rot[new_mask] = 0
        
        #show for debug
        cv2.imshow("rotate", img_rot)
        cv2.waitKey(0)
        
        #crop rectangle (atm it's black because of mask, change order later)
        img_crop = cv2.getRectSubPix(img_rot, (int(w),int(h)), (int(x),int(y)))
        
        #show for debug
        cv2.imshow("cut", img_crop)
        cv2.waitKey(0)
        
        #try to undo the roatation (failed atm)
        if angle > 0:
            angle = 360-angle
        else:
            angle = -360 - angle
        M = cv2.getRotationMatrix2D((x,y), angle, 1)
        img_rot = cv2.warpAffine(img_rot, M, (img_rot.shape[1], img_rot.shape[0]))
        
        #show for debug
        cv2.imshow("rotate back", img_rot)
        cv2.waitKey(0)
        
        
#####################################################################################################################################################
#
# call main
#
#####################################################################################################################################################        

    if __name__ == "__main__":
       main() 

finally:
    cv2.destroyAllWindows()
    print('done')