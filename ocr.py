# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:32:32 2021

@author: Simon
"""
import pytesseract
import cv2
from contours import *


def main():
    #main function for direct testing and comparison of OCR methods, for debug only
    
    filePaths, fileNames = searchFiles('.png', 'recttest')
    images = []
    
    #images = [cv2.imread(files, cv2.IMREAD_GRAYSCALE) for files in filePaths]
    images = [cv2.imread(files) for files in filePaths]
    for i, img  in enumerate(images):
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = normalizeImage(img)
        imga = new_preprocessing(img)
        imgb = preprocessing(img)
        
        text, rotate = image_to_text(imga)
        if rotate == True:
            imga = cv2.rotate(imga, cv2.cv2.ROTATE_180)
        #write text on image
        cv2.putText(imga, text, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        output('recttest_out', imga, fileNames[i],'new')
        
        text, rotate = image_to_text(imgb)
        if rotate == True:
            imgb = cv2.rotate(imgb, cv2.cv2.ROTATE_180)
        #write text on image
        cv2.putText(imgb, text, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        output('recttest_out', imgb, fileNames[i],'old')    

            
#####################################################################################################################################################
#
# Calls the ocr subfunctions. Returns processed image with text on it
#
#####################################################################################################################################################     

def ocr(img):
    #preimg = preprocessing(img)
    preimg = new_preprocessing(img)
    text, rotate = image_to_text(preimg)  
    if rotate == True:
        preimg = cv2.rotate(preimg, cv2.cv2.ROTATE_180)
    #write text on image
    cv2.putText(preimg, text, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return(preimg)

#####################################################################################################################################################
#
# OCR with pytesseract
#
#####################################################################################################################################################

def image_to_text(img):
    
    #call for pytesseract
    pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    #set default value for rotate flag
    rotate = False
    
    #try to read
    texta = pytesseract.image_to_string(img, config='board')
    #rotate and try again
    img = cv2.rotate(img, cv2.cv2.ROTATE_180)
    textb = pytesseract.image_to_string(img, config='board')
    
    #take the version with more chars detected and send them to textsplit for proper text output, set rotate to True if necessary
    if len(texta) >= len(textb):
        text = textsplit(texta)
    else:
        text = textsplit(textb)
        rotate = True
        
    #prints for debug
    print (text)
    print("img to text done")
    return(text, rotate)

def textsplit(text):
    #rearrange text to avoid crash
    arr = text.split('\n')[0:-1]
    text = '\n'.join(arr)
    return(text)

#####################################################################################################################################################
#
# create binary images for ocr
#
#####################################################################################################################################################

def preprocessing(img):
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # multiple blurring and normalization to get better contours
    for i in range (10):
            
        blur = cv2.medianBlur(gray, 3)
 
        gray = normalizeImage(blur)
            
        # set everything lower than 50 to 0
        #gray = np.where(gray < 60, 0, gray)
            
        if i % 10 == 0:
                
            gray = cv2.fastNlMeansDenoising(gray,7,7,7)
    gray = (255-gray)
    return(gray)

def new_preprocessing(img):
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = normalizeImage(gray)    
    # multiple blurring and normalization to get better contours
    #for i in range (10):
            
        #blur = cv2.medianBlur(gray, 3)
        #blur = cv2.bilateralFilter(gray,9,75,75)
        #gray = normalizeImage(blur)
            
        # set everything lower than 50 to 0
        #gray = np.where(gray < 60, 0, gray)
            
        #if i % 10 == 0:
                
    #gray = cv2.fastNlMeansDenoising(gray,15,15,15)
    gray = cv2.bilateralFilter(gray,9,100,100)  
    gray = cv2.fastNlMeansDenoising(gray,15,15,15)
    #gray = cv2.medianBlur(gray, 15)
    gray = (255-gray)
    gray = normalizeImage(gray)
    gray = sharp_kernel(gray)

    return(gray)

def sharpening(img):
    #laplacian of gaussian:
    #variables for laplace
    ddepth = cv2.CV_16S
    kernel_size = 3
    #gaussian
    blur = cv2.GaussianBlur(img, (3,3), 0)
    #laplacian
    laplace = cv2.Laplacian(blur, ddepth, ksize = kernel_size)
    conv = cv2.convertScaleAbs(laplace)
    #substraction
    
    
    #cv2.imshow("laplace", conv)
    #cv2.waitKey()
    
    return(dst)

def sharp_kernel(img):

    kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, 0]], np.float32)
    #kernel = 1/3 * kernel
    dst = cv2.filter2D(img, -1, kernel)
    dst = normalizeImage(dst)

    return(dst)
    
if __name__ == "__main__":
    main() 
