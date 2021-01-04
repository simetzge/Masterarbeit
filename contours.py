# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:39:38 2020

@author: Simon
"""

import numpy as np
import cv2
import os
import re
#from ocr import *

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
        print(len(rois))    
        
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
            print(len(rois))    
        
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
            
            print(roi["same"]) 
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
        gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
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

                    print ("asra " + str(asra))
                    
                    #ignore this shape if aspect ratio doesn't fit
                    if not (asra < aspectRatio * 1.4 and asra > aspectRatio *0.6):
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
        yb = min(recta[1] + recta[3],recta[1] + rectb[3])
        
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
            
            #(x, y), (w, h), angle = rect
            bl, br, tr , tl = cv2.boxPoints(rect).astype('int32')
            
            crop = img[min(tl[1],br[1]): max(tl[1],br[1]),min(tl[0],br[0]):max(tl[0],br[0])]
            
            #output('rectanglecut', rectcut, fileName) 
            output('rect', crop, fileName, str(i)) 
            
            mask[min(tl[1],br[1]): max(tl[1],br[1]),min(tl[0],br[0]):max(tl[0],br[0])] = True
            
        imgcut = img.copy()
        rectcut = imgcut[mask]
        imgcut[mask] = 0
        
        #send the modified images in the output function
        output('imagecut', imgcut, fileName)     

#####################################################################################################################################################
#
# call main
#
#####################################################################################################################################################        

    if __name__ == "__main__":
       main() 

finally:
    print('done')