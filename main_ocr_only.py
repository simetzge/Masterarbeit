# -*- coding: utf-8 -*-
"""
Created on Sun JUN 20 12:39:38 2021

@author: Simon Metzger

licensend under Attribution-NonCommercial-ShareAlike 3.0 Germany

CC BY-NC-SA 3.0 DE
"""

import cv2
from basics import *
from evaluation import *
from flags import *
from ocr import *

try:   
#####################################################################################################################################################

    def main():
    
        indiOCR(folder = "crop adaptive simple")
        #indiOCR(folder = "crop adaptive hough")
        #indiOCR(folder = "crop iterative simple")
        #indiOCR(folder = "crop iterative hough")
        #indiOCR(folder = "hough only")
        #indiOCR(folder = "simple only")
        #indiOCR(folder = "adaptive only")
        #indiOCR(folder = "iterative only")
        #indiOCR(folder = "crop")
        #indiOCR(folder = "collection")

    def indiOCR(folder = "crop"):
        #main function for direct testing and comparison of OCR methods, for debug only
        filePaths, fileNames = searchFiles(INPUT_FORMAT, folder)
        
        #open the files in cv2
        images = []
        cropImages = []
        sameImages = []
        cropNames = []
        ocrlist =[]
        fileNames = [(fileName[0:-6] + ".JPG") for fileName in fileNames]
        fileNames.append("end")
        #images = [cv2.imread(files, cv2.IMREAD_GRAYSCALE) for files in filePaths]
        images = [cv2.imread(files) for files in filePaths]
           
        for i, img  in enumerate(images):
            if CHECK_PICTURE != "":
                if not CHECK_PICTURE in fileNames[i]:
                    continue   
            if fileNames[i] == fileNames[i+1]:
                sameImages.append(img)
            else:
                sameImages.append(img)
                cropImages.append(sameImages)
                cropNames.append(fileNames[i])
                sameImages = []

        for j, crops in enumerate(cropImages):
            print (str(j) + "/" + str(len(cropImages)))
            ocrlist.append(ocr(crops, cropNames[j]))
    
        #ocrOutput(ocrlist)
        evaluation = csvInput(EVALUATION_LIST)
        compared = comparison(ocrlist)
        evaluated = evaluate(evaluation, compared)
        csvOutput(evaluated)
    
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