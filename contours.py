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

#####################################################################################################################################################
#
# flags
#
##################################################################################################################################################### 

IMG_TARGET_SIZE = 1000
THRESHOLD_MIN = 90
THRESHOLD_MAX = 255
MODIFY_THRESHOLD = False
USE_TEMPLATE = True
USE_ABSOLUTE_PATH = True
ABSOLUTE_PATH = "C:\\Users\\Simon\\Desktop\\masterarbeit\\contours"

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
            
            if 'template' in fileNames[i]:
                continue
            
            if MODIFY_THRESHOLD:
                rect_detect_iterative(img, fileNames[i])
            else:
                rect_detect_adaptive(img, fileNames[i])
        

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
        
        cut(img, rois, fileName)
    
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
            roisImg = cv2.drawContours(img, contours, -1, (0, 0, 230))
        
            #add the found rectangles in green to image
            roisImg = cv2.drawContours(roisImg, [cv2.boxPoints(rect).astype('int32') for rect in rois], -1, (0, 230, 0))
        
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
        
        mask = np.zeros(img.shape[:2], dtype=bool)
        #hier weitermachen rects stimmen nicht ganz und müssen korrigiert werden. das resultiert aus der Rotation der Rechtecke. cv.boxpoints
        #exrahiert die Eckpunkte nach beschriebenem Schema, der Winkel ist ebenfalls bekannt. Problem ist, wie kriege ich das Rechteck sauber
        #aus dem eigentlichen Bild? umgekehrt, die Tafel auszuschneiden und zu drehen scheint nicht das Problem (link)
        
        for i, rect in enumerate(rects):
            
            (x, y), (w, h), angle = rect
        
            bl, br, tr , tl = cv2.boxPoints(rect).astype('int32')
            
            #crop = img[min(tl[1],br[1]): max(tl[1],br[1]),min(tl[0],br[0]):max(tl[0],br[0])]
            
            #trying new version
            crop =hough_rotate(img,rect)
            
            #old version, works, but not perfect
            #crop = rotate_board (img, rect)
            
            #output('rectanglecut', rectcut, fileName)
            crop = preprocessing (crop)
            
            image_to_text(crop)
            
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

    def preprocessing(img):
        
        img = cv2.GaussianBlur(img,(3,3),5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #ret, img = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY,cv2.THRESH_OTSU) #imgf contains Binary image
        #img = scaleImage(img)
        #filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
        
        kernel = np.ones((1,1),np.uint8)
        
        #img = cv2.erode(img,kernel,iterations = 10)
        
        openening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        
        closing = cv2.morphologyEx(openening, cv2.MORPH_CLOSE, kernel)
        
        ret, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
        img = cv2.GaussianBlur(img,(1,1),0)
        #ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        img = cv2.GaussianBlur(img,(1,1),0)
        
        #ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY  + cv2.THRESH_OTSU)
        img = cv2.GaussianBlur(img,(1,1),0)
        #img = cv2.bitwise_or(img, closing)
        img = cv2.GaussianBlur(img,(1,1),0)
        #img = cv2.medianBlur(img,5) 
               
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
        
        return(img)
    
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
        dst = np.array([[0, h-1],[0, 0],[w-1, 0],[w-1, h-1]], dtype="float32")
        
        #get rotation matrix
        M = cv2.getPerspectiveTransform(src, dst)
        
        #warp
        warped = cv2.warpPerspective(img, M, (int(w), int(h)))
        
        return (warped)

#####################################################################################################################################################
#
# better crop with Hough-Transform
#
#####################################################################################################################################################

    def hough_rotate(img, rect):
        
        
        #how much bigger the crop image is than the board
        sizeFactor = 0.2     
        (x, y), (w, h), angle = rect
        
        #crop image with a larger area than the detected rect to get the corners of the board
        bl, br, tr , tl = cv2.boxPoints(rect).astype('int32')
        crop_img = img[int(min(tl[1],br[1]) - sizeFactor * w): int(max(tl[1],br[1]) + sizeFactor * w),int(min(tl[0],br[0]) - sizeFactor * h):int(max(tl[0],br[0]) + sizeFactor * h)]
        
        #preprocessing: scale, blur, grayscale, normalize, binary threshold 180, blur, skeleton, blur
        crop_img = scaleImage(crop_img)        
        blur = cv2.GaussianBlur(crop_img,(3,3),5)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)        
        gray = normalizeImage(gray)        
        ret, binary = cv2.threshold(gray, 180, THRESHOLD_MAX, cv2.THRESH_BINARY)
        binary = cv2.GaussianBlur(binary,(3,3),5)           
        binary = skeleton (binary)
        binary = cv2.GaussianBlur(binary,(3,3),5)
        
        #get shape
        height, width = binary.shape 
        
        # cannyedge        
        dst = cannyThreshold(binary)
        #hough with canny edge
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
        
        #cdst = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)  
        
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
                #cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
                # add lines to List
                line =[pt1,pt2]
                lineList.append(line)                
                
            # calculate every intersection between lines 
            for i in range(0, len(lineList)):    
                for j in range(0, len(lineList)):
                    # skip intersection of line with itself
                    if lineList[i] == lineList[j]:
                        break
                    
                    #skip if lines are in the same direction
                    quia = getQuadrant(binary, lineList[i][0])
                    quib = getQuadrant(binary, lineList[i][1])
                    quja = getQuadrant(binary, lineList[j][0])
                    qujb = getQuadrant(binary, lineList[j][1]) 
                    if (quia == quja or quia == qujb) and (quib == quja or quib == qujb):
                        break
                    
                    # call intersection calculation
                    inter = intersection(lineList[i], lineList[j])
                    #ignor if the intersection is on the corners
                    if (inter[0] < 0 or inter [0] > max(width,height)) or (inter[1] < 0 or inter[1] > max(width,height)):
                        break
                    interList.append(inter)
                    # add intersections as dots to output image for visualization
                    #cdst = cv2.circle(cdst, interList[-1], 4, (255,0,255), 4)
        
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
        tl = list(getCorner(tlList))
        tr = list(getCorner(trList))
        bl = list(getCorner(blList))
        br = list(getCorner(brList))
        
        #put points in array
        src = [bl, tl, tr, br]
        #get array for destination
        dst = np.array([[0, height-1],[0, 0],[width-1, 0],[width-1, height-1]], dtype="float32")
        
        #get rotation matrix
        M = cv2.getPerspectiveTransform(np.float32(src), dst) 
        #warp
        warped = cv2.warpPerspective(crop_img, M, (int(width), int(height)))
        
        if warped.shape[0] > warped.shape[1]:
            #warped = np.rot90(warped)
            warped = cv2.rotate(warped, cv2.cv2.ROTATE_90_CLOCKWISE) 
        # visualization for debug
        #cdst = crop_img
        #cdst = cv2.drawContours(cdst, [cv2.boxPoints(((tl[0], tl[1]), (10, 10), 0)).astype('int32')], -1, (250, 0, 250))
        #cdst = cv2.drawContours(cdst, [cv2.boxPoints(((tr[0], tr[1]), (10, 10), 0)).astype('int32')], -1, (250, 0, 250))
        #cdst = cv2.drawContours(cdst, [cv2.boxPoints(((bl[0], bl[1]), (10, 10), 0)).astype('int32')], -1, (250, 0, 250))
        #cdst = cv2.drawContours(cdst, [cv2.boxPoints(((br[0], br[1]), (10, 10), 0)).astype('int32')], -1, (250, 0, 250))   
        #cv2.imshow("test", warped)
        #cv2.waitKey()
        
        return (warped)
        
#####################################################################################################################################################
#
# get corner
#
#####################################################################################################################################################

    def getCorner(inList):
        
        iouList = []
        for i in range(len(inList)):
            iou = 0
            for j in range(len(inList)):
                if inList[i] != inList[j]:
                    recta = inList[i][0], inList[i][1],10,10
                    rectb = inList[j][0], inList[j][1],10,10
                    if intersection_over_union(recta, rectb) > 0.9:
                        iou += 1
            iouList.append(iou)
        position = np.argsort(iouList)
        corner = inList[position[-1]]
        #corner = [float(corn) for corn in corner]
        return(corner)
    
#####################################################################################################################################################
#
# quadrant check
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