# -*- coding: utf-8 -*-
"""
Created on Sun JUN 20 12:39:38 2021

@author: Simon Metzger

licensend under Attribution-NonCommercial-ShareAlike 3.0 Germany

CC BY-NC-SA 3.0 DE
"""

#from flags import *
import ocr
#from cnn import *
import evaluation
import basics
import rectdetection
import cv2
import time

try:   
#####################################################################################################################################################
    
    def main():
        #start time measurement
        start = time.process_time()
        
        #load settings this is so important i'm so glad it works finally
        basics.loadSettings()
        
        #get paths and names of all images in folder input
        filePaths, fileNames = basics.searchFiles(basics.INPUT_FORMAT, 'input')
        #open the files in cv2
        #images = []
        ocrlist =[]        
        #images = [cv2.imread(files) for files in filePaths]              
        #get aspect ratio from template if flag is set        
        if basics.USE_TEMPLATE == True:
            rectdetection.getAspectRatio(filePaths, fileNames)
            print("template check")            
        #detect rectangles in every image, adaptive or iterative
        #for i, img  in enumerate(images):
            
        #get number of files to track the progress
        file_number = len(fileNames)
        for i , file in enumerate(filePaths):
            img = cv2.imread(file)
            #skip template
            if 'template' in fileNames[i]:
                file_number = file_number -1
                continue
            #skip all pictures but the one that should be checked
            if basics.CHECK_PICTURE != "":
                if not basics.CHECK_PICTURE in fileNames[i]:
                    continue            
            #print progress based on the number of files to be processed                
            print("The next image is \"" + fileNames[i] + "\" (" + str(i) + "/" + str(file_number) + ")")            
            #run iterative or adaptive detection, depending on settings    
            if rectdetection.MODIFY_THRESHOLD:
                rects,conts = rectdetection.rect_detect_iterative(img, fileNames[i])
            else:
                rects,conts = rectdetection.rect_detect_adaptive(img, fileNames[i])
                rects = [rectdetection.rescaleRect(img, rect) for rect in rects]
                #oldconts = conts
                conts = [rectdetection.getCont(img,rect) for rect in rects]
                #contCompare(conts, oldconts, img)
            if len(rects) > 0:
                #crop found rectangle
                if basics.CONT_BASED_CUT == True:
                    masked = rectdetection.contMask(img,conts)
                else:
                    masked = None
                cropimgs, restimg = rectdetection.cut(img, rects, masked = masked)
                #perform OCR on cropped rectangles if flag is set
                if basics.OCR:
                    ocrlist.append(ocr(cropimgs, fileNames[i]))
                
                    #just print the crops if OCR flag is not set
                for j, crop in enumerate(cropimgs):
                    basics.output('crop', crop, fileNames[i], str(j))
                
                #write images without rectangles
                basics.output('imagecut', restimg, fileNames[i])
            #CNN PART REMOVED
            #use cnns to identify objects in the images
            #if USE_CNN == "coco" or USE_CNN == "both":
            #    output('coco', coco(img), fileNames[i])
            #if USE_CNN == "yolo" or USE_CNN == "both":
            #    output('yolo', yolo(img), fileNames[i])
        if basics.OCR: ocr.ocrOutput(ocrlist)
        #csvOutput(outputlist)
        if basics.EVALUATE:
            if basics.OCR:
                evalList = basics.csvInput(basics.EVALUATION_LIST)
                compared = evaluation.comparison(ocrlist)
                evaluated = evaluation.evaluate(evalList, compared)
                basics.csvOutput(evaluated)
            else:
                print("No OCR no evaluation")                
        #print time measurement
        run_time = time.process_time() - start
        print('Run complete!')
        print("The detection took " + str(run_time) + " seconds, " + str(run_time / file_number) + " on average.")

#####################################################################################################################################################
#
# call main
#
#####################################################################################################################################################            

    if __name__ == "__main__":
       main() 
       
finally:
    cv2.destroyAllWindows()
    