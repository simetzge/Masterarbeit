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
        
        #get paths and names of all images in folder input
        filePaths, fileNames = searchFiles('.jpg', 'input')
    
        #open the files in cv2
        images = []
        #images = [cv2.imread(files, cv2.IMREAD_GRAYSCALE) for files in filePaths]
        images = [cv2.imread(files) for files in filePaths]
        
        #scale images to 1000px
        images = [scaleImage(img) for img in images]
        
        #get aspect ratio from template if flag is set
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
            
            #crop found rectangles
            cropimgs, restimg = cut(img, rects)
            #perform OCR on cropped rectangles
            for j, crop in enumerate(cropimgs):
                ocrimg = ocr(crop)
                output('rect', ocrimg, fileNames[i], str(j))
                #output('rect', crop, fileNames[i], str(j))
            
            #write images without rectangles
            output('imagecut', restimg, fileNames[i])
            #CNN(img)
        print(COUNTER)


#####################################################################################################################################################
#    
# function for searching all files with the matching extension in the input directory
# will return the paths and names of all files found
#
#####################################################################################################################################################

    def searchFiles(extension,folder):
        
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
        if folder in dirs:
            print(folder + ' gefunden')
            
            #list all files in input dir
            content = os.listdir(path + '\\' + folder)
            
            #match the files with given extension
            for item in content:
                jregex = re.compile(extension, re.IGNORECASE)
                match = jregex.search(item)
                #if found add to file array
                if match != None:
                    files.append(path + '\\' + folder + '\\' + item)
                    names.append(item)
        #print note and end skript if no input dir
        else:
            print(folder + ' fehlt')
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
        #write files, add name modification if necessary
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
        return (cv2.resize(img, (0,0), fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC))

#####################################################################################################################################################
#      
# normalize grayscale image to range from 0 to 255
#
#####################################################################################################################################################
    
    def normalizeImage(img):
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
        
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        #add contours in red to image
        roisImg = cv2.drawContours(gray, contours, -1, (0, 0, 230))
        
        #add the found rectangles in green to image
        roisImg = cv2.drawContours(roisImg, [cv2.boxPoints(rect).astype('int32') for rect in rois], -1, (0, 230, 0))
        
        #send the modified images in the output function
        output('output', roisImg, fileName, 'adaptive')

        return(rois)
    
#####################################################################################################################################################
#
# sets an increasing threshold, sends the results to rect_detect and those results to output
#
#####################################################################################################################################################

    def rect_detect_iterative(img, fileName):
        
        thresh = THRESHOLD_MIN
        allRois = []
        
        #convert to grayscale and normalize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
        gray = normalizeImage(gray)
        
        #search for rectangles with increasing threshold, max 200
        while thresh <= 200:
            
            rois = []            
            contours = []            
            
            ret, binary = cv2.threshold(gray, thresh, THRESHOLD_MAX, cv2.THRESH_BINARY)    
            
            contours, rois = rect_detect(binary)
            
            if len(rois) > 0:       
                
                allRois.append(rois)     
                
            thresh += 5
        
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
        #add contours in red to image
            if roi["same"] >= 6:              
                #roisImg = cv2.drawContours(gray, contours, -1, (0, 0, 230))
                rect = (roi["x"],roi["y"]),(roi["w"],roi["h"]),roi["angle"]
                rects.append(rect)
                same = roi["same"]
        #convert to colored img for output
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        #add the found rectangles in green to image
        roisImg = cv2.drawContours(gray, [cv2.boxPoints(rect).astype('int32') for rect in rects], -1, (0, 230, 0))
                    
        #send the modified images in the output function
        output('output', roisImg, fileName, str(same))
        
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

        return(iou)

#####################################################################################################################################################
#
# cuts rectangle from image
# returns both, modified image and rectangle
#
#####################################################################################################################################################

    def cut(img, rects):
        
        crops = []
        # generate mask for extraction of rectangles
        mask = np.zeros(img.shape[:2], dtype=bool)
        
        # crop every rectangle with simple crop ()
        for i, rect in enumerate(rects):
            
            (x, y), (w, h), angle = rect
        
            bl, br, tr , tl = cv2.boxPoints(rect).astype('int32')
            
            if SIMPLE_CROP:
                #old version, works, but not perfect
                crop = rotate_board (img, rect)
            else:
                # new version
                crop = hough_rotate(img,rect, CUT_THRESH)
            
            #end function if no crop image found (hough rotate returns [None] if something went wrong)
            if len(crop) < 2:
                continue
            
            crops.append(crop)
            
            # mask area sligtly bigger than detected rect to cut the complete board with its border
            mask[int(min(tl[1],br[1]) - 0.1 * w): int(max(tl[1],br[1]) + 0.1 * w),int(min(tl[0],br[0]) - 0.1 * h):int(max(tl[0],br[0]) + 0.1 * h)] = True
        
        #modify image: set mask area to black
        imgcut = img.copy()
        rectcut = imgcut[mask]
        imgcut[mask] = 0
        
        return(crops, imgcut)   

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
        if USE_TEMPLATE == True and 'aspectRatio' in globals():
            dsize = (IMG_TARGET_SIZE, int(IMG_TARGET_SIZE / aspectRatio))
        else:
            dsize = (IMG_TARGET_SIZE, int(IMG_TARGET_SIZE * 0.8))

        # resize image
        warped = cv2.resize(warped, dsize, interpolation = cv2.INTER_CUBIC)
        
        return (warped)

#####################################################################################################################################################
#
# better crop with Hough-Transform
#
#####################################################################################################################################################

    def hough_rotate(img, rect, threshold):
        
        #debug flag
        debug_hough = False
        
        #if threshold is too low, use simple crop
        if threshold < 100:
            return (rotate_board(img, rect))
        
        (x, y), (w, h), angle = rect

        new_rect = (x,y), (int(w*1.3), int(h*1.3)), angle

        crop_img = rotate_board(img, new_rect)
        
        if crop_img.shape[0] > crop_img.shape[1]:
            #warped = np.rot90(warped)
            crop_img = cv2.rotate(crop_img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        #preprocessing: scale, blur, grayscale, normalize, binary threshold 180, blur, skeleton, blur
        #crop_img = scaleImage(crop_img)
        blur = cv2.bilateralFilter(crop_img,9,75,75)
        blur = cv2.fastNlMeansDenoising(blur,7,7,15)        
        blur = cv2.GaussianBlur(blur,(7,7),15)
        
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)       
        gray = normalizeImage(gray)        
        mean = np.mean(gray)
        #mean *1.2 bisher zweitbeste (31), beste mean+25 (27)
        
        ret, binary = cv2.threshold(gray, int(mean+30), THRESHOLD_MAX, cv2.THRESH_BINARY)
        
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
                    
                    #if not ((inter[0] < 1 or inter [0] > width) or inter[1] < 1 or inter[1] > height):
                        
                    heightdiff = int((height - height / 1.1) / 2)
                    widthdiff = int((width - width / 1.1) / 2)
                    if not ((inter[0] < widthdiff or inter [0] > width-widthdiff) or inter[1] < heightdiff or inter[1] > height - heightdiff):    
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
        warped = cv2.resize(warped, dsize, interpolation = cv2.INTER_CUBIC)
        
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
        n = iouList.count(int(iouList[position[-1]]))
        #print (str(iouList[position[-1]]) + " kommt " + str(n) + " mal vor")
        
        #get mean of coordinates
        corner = inList[position[-1]]
        cornarray = []
        #cornarray = inList[position[-n:0]]
        for i in range(n):
            cornarray.append(inList[position[-i]])

        corner = (int(sum(c[0] for c in cornarray)/n),int(sum(c[1] for c in cornarray)/n))
        #corner = np.mean(cornarray, axis = 0)
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