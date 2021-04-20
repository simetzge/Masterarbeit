# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:39:38 2020

@author: Simon
"""

import numpy as np
import cv2
import math
from flags import *
from ocr import *
from cnn import *
from evaluation import *
from basics import *

try:   
#####################################################################################################################################################
    
    def main():
        
        #get paths and names of all images in folder input
        filePaths, fileNames = searchFiles('.jpg', 'input')
    
        #open the files in cv2
        images = []
        
        ocrlist =[]
        
        #images = [cv2.imread(files, cv2.IMREAD_GRAYSCALE) for files in filePaths]
        images = [cv2.imread(files) for files in filePaths]
        
        #scale images to 1000px -> moved to rect_detect_adaptive
        #images = [scaleImage(img) for img in images]
        
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
                rects,conts = rect_detect_iterative(img, fileNames[i])
            else:
                rects,conts = rect_detect_adaptive(img, fileNames[i])
                rects = [rescaleRect(img, rect) for rect in rects]
                #oldconts = conts
                conts = [getCont(img,rect) for rect in rects]
                #contCompare(conts, oldconts, img)

            if len(rects) > 0:
                #crop found rectangle
                if CONT_BASED_CUT == True:
                    masked = contMask(img,conts)
                else:
                    masked = None
                cropimgs, restimg = cut(img, rects, masked = masked)
                #perform OCR on cropped rectangles if flag is set
                imagedict = {"image": fileNames[i]}
                if OCR:
                    bestguess = ""
                    for j, crop in enumerate(cropimgs):
                        textimg, text = ocr(crop)
                        boximg, boxtext = ocr(crop, mode = 'image_to_box')
                        #delete spaces, check if box or txt are longer than bestguess, replace bestguess in this case
                        box = boxtext.replace(" ", "")
                        txt = text.replace(" ", "")
                        if max(len(box),len(txt)) > len(bestguess):
                            if len(box) > len(txt):
                                bestguess = boxtext
                            else:
                                bestguess = text
                        #output of rectimage and boximage
                        output('rect', textimg, fileNames[i], str(j))
                        output('box', boximg, fileNames[i], str(j))
                        output('crop', crop, fileNames[i], str(j))
                        #output('rect', crop, fileNames[i], str(j))
                        #add rectdict to imagedict and imagedict to ocrlist
                        rectdict = {"rectangle": fileNames[i] + "_" + str(j), "textimage": text, "boximage": boxtext}
                        imagedict["rectangle " + str(j)] =  rectdict
                        imagedict["bestguess"] = bestguess
                    
                    ocrlist.append(imagedict)
                else:
                    #just print the crops if OCR flag is not set
                    for j, crop in enumerate(cropimgs):
                        output('crop', crop, fileNames[i], str(j))
                
                #write images without rectangles
                output('imagecut', restimg, fileNames[i])
            
            
            #use cnns to identify objects in the images
            if USE_CNN == "coco" or USE_CNN == "both":
                output('coco', coco(img), fileNames[i])
            if USE_CNN == "yolo" or USE_CNN == "both":
                output('yolo', yolo(img), fileNames[i])
        
        #csvOutput(outputlist)
        if EVALUATE:
            evaluation = csvInput("evaluationlist.csv")
            compared = comparison(ocrlist)
            evaluated = evaluate(evaluation, compared)
            csvOutput(evaluated)
        print(COUNTER)

    
##################################################################################################################################################### 
#
# sets an adaptive threshold, sends the results to rect_detect and those results to output
#
#####################################################################################################################################################

    def rect_detect_adaptive(img, fileName):
        
        #convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        gray = normalizeImage(gray)
        
        gray = cv2.bilateralFilter(gray,9,9,9)  

        #scale image
        scaled = scaleImage(gray)
        
        #set threshold
        binary = cv2.adaptiveThreshold(scaled,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,1)       
        
        #detect contours and rectangles
        contours, rois, roisConts = rect_detect(binary)
        
        #rois = [rescaleRect(max(gray.shape[0],gray.shape[1]), roi) for roi in rois]
        
        scaled = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
        
        #add contours in red to image
        roisImg = cv2.drawContours(scaled, contours, -1, (0, 0, 230))
        
        #rescale rois
        #scaledrois = [rescaleRect(gray, rect) for rect in rois]
        
        #add the found rectangles in green to image
        roisImg = cv2.drawContours(scaled, [cv2.boxPoints(rect).astype('int32') for rect in rois], -1, (0, 230, 0), 2)
        
        #send the modified images in the output function
        output('output', roisImg, fileName, 'adaptive')
        #rescaleCont(img, scaled, rectConts)
        #return(scaledrois)
        return(rois, roisConts)
    
    #rescale rectangles based on the size of the original image and the flag IMG_TARGET_SIZE which is used to downscale images at first
    def rescaleRect(img, rect):
        scale = np.max(img.shape) / IMG_TARGET_SIZE
        (x,y), (w,h), angle = rect
        rect = (x*scale, y*scale), (w*scale, h*scale), angle
        return (rect)
    
##################################################################################################################################################### 
#
# input: image and rectangle within this image
# ouput: largest contour in the rectangle
# purpose: find the contour that led to the rectangle again, necessary when the scaling changed
#
#####################################################################################################################################################
   
    def getCont(img, rect):
        #debug flag
        debugCont = False
        #turn rect into mask and remove everything outside of this area
        bl, br, tr , tl = cv2.boxPoints(rect).astype('int32')
        (x,y),(w,h),angle = rect
        mask  = np.zeros(img.shape,np.uint8)
        cv2.drawContours(mask,[cv2.boxPoints(rect).astype('int32')],0,(255,255,255),-1)
        mask = cv2.bitwise_not(mask)
        newImg = img.copy()
        newImg = cv2.bitwise_and(cv2.bitwise_not(mask), newImg)
        #turn the modified image into grayscale and binarize it
        gray = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)
        #norm = normalizeImage(gray)
        binary  = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,0)
        #apply findContours
        contourList, hierarchy  = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #set variables for maximum values on default
        maxArea =  0
        maxCont = None
        #iterate through contourList and get the areas
        for i, contour in enumerate(contourList):
            contArea = cv2.contourArea(contour)
            #ignore contours larger than the rectangle
            if contArea > w * h:
                continue
            #(x, y), (w, h), angle = rect = cv2.minAreaRect(contour)
            #rectArea = w * h
            #if rectArea  == 0:
             #   continue
         
            #save area and contour if the area is larger than the previous maximum
            if contArea > maxArea:
                maxArea = contArea
                maxCont = contour
        #show the contour and it's cropped version for debug reasons
        if debugCont:
            show(gray)
            show(binary)
            newImg = cv2.drawContours(img, maxCont, -1, (0, 0, 230),2)
            show(newImg)
            cropImgs = contMask(img, [maxCont], rescale = False)
            for crop in cropImgs:
                show(crop)
        #return largest contour
        #maxCont = cv2.convexHull(maxCont)
        #epsilon = 0.1*cv2.arcLength(maxCont,True)
        #maxCont = cv2.approxPolyDP(maxCont,epsilon,True)
        return(maxCont)
        
# gets an imagen and contours, returns an image where everything but the contour area is set to 0   
    def contMask(img,contours, rescale = False):
        cropImgs = []
        if rescale == True:    
            scaledImg = scaleImage(img)
            for contour in contours:
                mask = np.zeros(scaledImg.shape,np.uint8)
                cv2.drawContours(mask,[contour],0,(255,255,255),-1)
                #scaledArea = cv2.bitwise_and(mask, scaledImg)
                scaledMask = scaleImage(mask, np.max(img.shape))
                cut = cv2.bitwise_and(scaledMask, img)
                cropImgs.append(cut)
        else:
            for contour in contours:
                mask = np.zeros(img.shape,np.uint8)
                #mask = img
             
                mask[:,:,:]=0
             
                cv2.drawContours(mask,[contour],0,(255,255,255),-1)
                #show(mask)
                cut = cv2.bitwise_and(mask, img)
                #show(cut)
                cropImgs.append(cut)
        return(cropImgs)    
            
#####################################################################################################################################################
#
# sets an increasing threshold, sends the results to rect_detect and those results to output
#
#####################################################################################################################################################

    def rect_detect_iterative(img, fileName):
        
        thresh = THRESHOLD_MIN
        allRois = []
        allConts = []
        
        #convert to grayscale and normalize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     
        gray = normalizeImage(gray)
        
        #search for rectangles with increasing threshold, max 200
        while thresh <= 200:
            
            rois = []            
            contours = []            
            
            ret, binary = cv2.threshold(gray, thresh, THRESHOLD_MAX, cv2.THRESH_BINARY)    
            
            contours, rois, rectConts = rect_detect(binary)#img for debug
            
            if len(rois) > 0:       
                
                allRois.append(rois)
                allConts.append(rectConts)
            thresh += 5
        
        #new rois list
        rois_list = []        
        #go through the found rectangles and add them to an array of dictionaries
        for i, r in enumerate(allRois):
            cont = allConts[i]
            for j in range(len(r)):
                (x,y), (w,h), angle = r[j]
                rois_dict = {
                    }
                rois_dict["x"] = x
                rois_dict["y"] = y
                rois_dict["w"] = w
                rois_dict["h"] = h
                rois_dict["angle"] = angle
                rois_dict["same"] = 0
                rois_dict["cont"] = cont
                
                
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
        cont = None
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
                cont = roi["cont"]
        #convert to colored img for output
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        #add contours in red to image
        #roisImg = cv2.drawContours(gray, contours, -1, (0, 0, 230))
        #add the found rectangles in green to image
        roisImg = cv2.drawContours(gray, [cv2.boxPoints(rect).astype('int32') for rect in rects], -1, (0, 230, 0),3)
                    
        #send the modified images in the output function
        output('output', roisImg, fileName, str(same))
        
        return(rects,cont)
        
#####################################################################################################################################################
#
# detetects all rectangles in a given binary image
#
#####################################################################################################################################################
        
    def rect_detect(binary):
        
        #findcontours
        contours, hierarchy  = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        #creat array for regions of interest
        rois = []
        rectConts = []
        
        #move through every contour in array contours
        for contour in contours:
            
            #compute contour area
            contArea = cv2.contourArea(contour)
            
            #throw out too small areas
            if not max(binary.shape[0],binary.shape[1]) < contArea:
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
            #contour = cv2.convexHull(contour)
            rectConts.append(contour)
        
        return (contours, rois, rectConts)
    
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
            img = scaleImage(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,1)

            #detect template
            contours, rois, rectConts = rect_detect(binary)
            
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

    def cut(img, rects, masked = None):
        
        crops = []
        # generate mask for extraction of rectangles
        mask = np.zeros(img.shape[:2], dtype=bool)
        
        # crop every rectangle with simple crop ()
        for i, rect in enumerate(rects):
            
            (x, y), (w, h), angle = rect
        
            bl, br, tr , tl = cv2.boxPoints(rect).astype('int32')
            
            if SIMPLE_CROP:
                #old version, works, but not perfect
                if masked != None:
                    crop = simple_crop (masked[i], rect)
                else:
                    crop = simple_crop (img, rect)
            else:
                # new version
                if masked != None:
                    crop = hough_rotate(masked[i],rect, CUT_THRESH)
                else:
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

    def simple_crop(img, rect):
        
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
        
        #if warped.shape[0] > warped.shape[1]:
        if h > w:
            #warped = np.rot90(warped)
            warped = cv2.rotate(warped, cv2.cv2.ROTATE_90_CLOCKWISE)
        
        # dsize
        #if USE_TEMPLATE == True and 'aspectRatio' in globals():
         #   dsize = (IMG_TARGET_SIZE, int(IMG_TARGET_SIZE / aspectRatio))
        #else:
         #   dsize = (IMG_TARGET_SIZE, int(IMG_TARGET_SIZE * 0.8))

        # resize image
        #warped = cv2.resize(warped, dsize, interpolation = cv2.INTER_CUBIC)
        
        #resizing while keeping the aspect ratio
        warped = scaleImage(warped)
        
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
            return (simple_crop(img, rect))
        
        (x, y), (w, h), angle = rect

        new_rect = (x,y), (int(w*1.3), int(h*1.3)), angle

        crop_img = simple_crop(img, new_rect)
        
        if crop_img.shape[0] > crop_img.shape[1]:
            #warped = np.rot90(warped)
            crop_img = cv2.rotate(crop_img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        #preprocessing: scale, blur, grayscale, normalize, binary threshold 180, blur, skeleton, blur
        #crop_img = scaleImage(crop_img)
        blur = cv2.bilateralFilter(crop_img,9,75,75)
        blur = cv2.fastNlMeansDenoising(blur,7,7,15)        
        blur = cv2.GaussianBlur(blur,(7,7),15)
        
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)       
        norm = normalizeImage(gray)    
        mean = np.mean(norm)
        if debug_hough:
            print("threshold ist " + str(threshold) + " mean ist " + str(np.mean(gray)))
        #mean *1.2 bisher zweitbeste (31), beste mean+25 (27)
        
        ret, binary = cv2.threshold(gray, int(mean+30), THRESHOLD_MAX, cv2.THRESH_BINARY)
        
        if debug_hough:
            cv2.imshow("test", binary)
            cv2.waitKey()
        binary = cv2.GaussianBlur(binary,(3,3),15)    
        if CONT_BASED_CUT == False:
            binary = skeleton (binary)
        binary = cv2.GaussianBlur(binary,(3,3),15)
        
        #get shape
        height, width = binary.shape
        
        # cannyedge        
        dst = cannyThreshold(binary)
        #hough with canny edge
        lines = cv2.HoughLines(dst, 1, np.pi / 180, threshold, None, 0, 0)
        #lines = cv2.HoughLinesP(dst, 1, np.pi / 180, threshold, 30,10)
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
            #return (simple_crop(img, rect))
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
# mask cut
#
#####################################################################################################################################################

    def maskcut(img,contour):
        img = scaleImage(img)
        mask = np.zeros(img.shape,np.uint8)
        cv2.drawContours(mask,[contour],0,(255,255,255),-1)
        #newmask = np.zeros(img.shape,np.uint8)
        #newmask = img[mask]
        mask = cv2.bitwise_and(mask, img)
        #masked = cv2.bitwise_and(img, img, mask=mask)
        out = mask
        cv2.imshow("rotate back", out)
        cv2.waitKey(0)
        
#####################################################################################################################################################
#
# input: new contours made by getCont, old contours made by rectDetect, image
# ouput: none
# purpose: draw both contours on the image, to compare the differences
#
#####################################################################################################################################################
        
    def contCompare(conts, oldconts, img):
        for i,contour in enumerate(conts):
            showimg = img.copy()
            showimg = cv2.drawContours(showimg, contour, -1, (0, 230, 0),2)
            showimg = cv2.drawContours(showimg, oldconts[i], -1, (230, 0, 0),2)
            show(showimg)
        
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