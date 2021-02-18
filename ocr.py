# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:32:32 2021

@author: Simon
"""
import pytesseract
import cv2
from contours import *
from PIL import Image
TESTFLAG = True


def main():
    
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

            
        

def testocr():
    print('ocr standing by')

def image_to_text(img):
    pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    rotate = False
    #text = pytesseract.image_to_string(r'C:\Users\Simon\Desktop\masterarbeit\contours\rect\testpaint.png')
    
    #img = preprocessing(img)
    #try to read
    texta = pytesseract.image_to_string(img, config='board')
    #rotate and try again
    img = cv2.rotate(img, cv2.cv2.ROTATE_180)
    textb = pytesseract.image_to_string(img, config='board')
    
    #take the version with more chars detected
    #if len(texta) >= len(textb):
     #   text = texta
    #else:
     #   text = textb
      #  roate = True
    #if 'campidoglio' is not on the board, it should be a chalk board, therefore use simpler charset
    #if not"CAMPIDOGLIO" in text:
        #text = pytesseract.image_to_string(img, config='board')
    #split text to insert linebreaks
    #if the number of detected characters is still lower than 5, the detection failed
    if len(texta) >= len(textb):
        text = textsplit(texta)
    else:
        text = textsplit(textb)
        rotate = True
    
    #if len(text) < 5:
        #text = "fail"

    print (text)
    print("img to text done")
    print (rotate)
    return(text, rotate)

def textsplit(text):
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
    gray = normalizeImage(gray)  
    gray = cv2.fastNlMeansDenoising(gray,15,15,15)
    #gray = cv2.medianBlur(gray, 15)
    gray = (255-gray)

    return(gray)

if __name__ == "__main__":
    main() 
