# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:32:32 2021

@author: Simon
"""
import pytesseract
import cv2
import numpy as np
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
        imgc = sharpening(imga)
        imgb = preprocessing(img)
        imgd = sharp_kernel(imga)
        imge = unsharp_mask(imga)
        text, rotate = image_to_text(imga)
        if rotate == True:
            imga = cv2.rotate(imga, cv2.cv2.ROTATE_180)
        #write text on image
        cv2.putText(imga, text, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        output('recttest_out', imga, "img " + str(i) + ".jpg",'new')
        
        text, rotate = image_to_text(imgb)
        if rotate == True:
            imgb = cv2.rotate(imgb, cv2.cv2.ROTATE_180)
        #write text on image
        #cv2.putText(imgb, text, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        output('recttest_out', imgb, "img " + str(i) + ".jpg",'old')
        text, rotate = image_to_text(imgc)
        cv2.putText(imgc, text, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        output('recttest_out', imgc, "img " + str(i) + ".jpg",'new_sharp')
        text, rotate = image_to_text(imgd)
        cv2.putText(imgd, text, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        output('recttest_out', imgd, "img " + str(i) + ".jpg",'new_kernel')
        
        
        #binary = cv2.adaptiveThreshold(imgb,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,1)
        ret, binary = cv2.threshold(imgb, 120, 255, cv2.THRESH_BINARY) 
        rois = []
        #findcontours
        #contours, rois = rect_detect(binary)
        contours, hierarchy  = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #create rectangle around contours
        global aspectRatio
        aspectRatio = 0
        for contour in contours:
            (x, y), (w, h), angle = rect = cv2.minAreaRect(contour)
            contArea = cv2.contourArea(contour)
            if not 1000 < contArea:
                continue  
        #compute area of this rectangle
            rectArea = w * h
        #compare the areas to each other, make sure they don't differ too much
            if contArea / rectArea < 0.85:
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
                                #ignore contours as big as the image
                if w < binary.shape[0] * 0.5 or h < binary.shape[1]*0.5:
                    continue
                w = 0.95*w
                h = 0.95 *h
                rect = (x, y), (w, h), angle
            rois.append(rect)
        
        gray = cv2.cvtColor(imgb, cv2.COLOR_GRAY2BGR)
        
        #add contours in red to image
        roisImg = cv2.drawContours(gray, contours, -1, (0, 0, 230))
        
        #add the found rectangles in green to image
        roisImg = cv2.drawContours(roisImg, [cv2.boxPoints(rect).astype('int32') for rect in rois], -1, (0, 230, 0))
        
        for rect in rois:
            img = rotate_board(img, rect)
            cv2.putText(img, text, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            output('recttest_out', img, "img " + str(i) + ".jpg",'old')
        
        cv2.imshow("roi", img)
        cv2.waitKey()
    cv2.destroyAllWindows()

            
#####################################################################################################################################################
#
# Calls the ocr subfunctions. Returns processed image with text on it
#
#####################################################################################################################################################     

def ocr(img):
    preimg = preprocessing(img)
    
    preimg = getinnerrect(preimg)
    
    preimg = new_preprocessing(preimg)
    
    
    preimg = normalizeImage(preimg)
    
    preimg = (255-preimg)
    
    
    text, rotate = image_to_text(preimg)  
    if rotate == True:
        preimg = cv2.rotate(preimg, cv2.cv2.ROTATE_180)
    #write text on image
    
    #convert to colored img for output
    preimg = cv2.cvtColor(preimg, cv2.COLOR_GRAY2BGR)
    
    #color the darkest areas for debug
    #preimg = np.where(preimg < 20, 255, preimg)
    #for columns in preimg:
        #for rows in columns:
            #if rows[0] == 255 and rows[1] == 255 and rows[2] == 255:
                #rows[0] = 0
                #rows[1] = 0
    
    
    cv2.putText(preimg, text, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 230, 0), 3)
    return(preimg)

def getinnerrect(img):
    
    if len(img.shape) < 3:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = img
    #binary = cv2.adaptiveThreshold(imgb,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,1)
    ret, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY) 
    rois = []  
    contAreas = []

    #findcontours
    contours, hierarchy  = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        #(x, y), (w, h), angle = rect = cv2.minAreaRect(contour)
        contArea = cv2.contourArea(contour)
        contAreas.append(contArea)
        
        #if not 1000 < contArea:
            #continue  
        #compute area of this rectangle
       # rectArea = w * h
        #compare the areas to each other, make sure they don't differ too much
        #if contArea / rectArea < 0.7:
            #continue
            #compute if area is not empty
        #if rectArea != 0:
               
            #if template is used, check for aspect ratio
            #if USE_TEMPLATE == True and aspectRatio != 0:
                    
                #get aspect ratios of rect and approx
             #   asra = max(w,h)/min(w,h)
                    
                #ignore this shape if aspect ratio doesn't fit
              #  if not (asra < aspectRatio * 1.3 and asra > aspectRatio *0.7):
               #     continue                 
                
            #else aspect ratio should be max 2:1
            #else:
                #make sure the aspect ratio is max 2:1
             #   if max(w,h) > 2 * min(w,h):
              #      continue
            #ignore too small contours
        #if w < binary.shape[0] * 0.6 or h < binary.shape[1] * 0.6:
         #   continue
        #if w > binary.shape[0] * 0.99 or h > binary.shape[1] * 0.99:
           # continue
        
        #w = 0.95*w
        #h = 0.95*h
        #newrect = (x, y), (w, h), angle

        #rois.append(newrect)
    positions = np.argsort(contAreas)
    position = positions[-1]
    contour = contours[position]
    innerrect = cv2.minAreaRect(contour)
    rois.append(innerrect)
        
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
    #add contours in red to image
    #roisImg = cv2.drawContours(out, contours, -1, (0, 0, 230))
    #add the found rectangles in green to image
    #roisImg = cv2.drawContours(roisImg, [cv2.boxPoints(rect).astype('int32') for rect in rois], -1, (0, 230, 0))
    #img = rotate_board(img, rect)
    #cv2.imshow("test", roisImg)
    #cv2.waitKey()
    
    
    newrect = rois[0]
    #get boxpoints
    (x, y), (w, h), angle = newrect
    box = cv2.boxPoints(((x, y), (int(0.95*w), int(0.95*h)), angle))
    box = np.int0(box)
    

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
        
    #warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    return (warped)
    
    

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
    #config = ('board')
    config = ("board -l dic --oem 1 --psm 7")
    texta = pytesseract.image_to_string(img, config=config)
    #rotate and try again
    img = cv2.rotate(img, cv2.cv2.ROTATE_180)
    textb = pytesseract.image_to_string(img, config=config)
    
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
        
    if len(img.shape) < 3:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # multiple blurring and normalization to get better contours
    for i in range (10):
            
        blur = cv2.medianBlur(gray, 3)
        #blur = cv2.GaussianBlur(img, (3,3), 1)
 
        gray = normalizeImage(blur)
            
        # set everything lower than 50 to 0
        #gray = np.where(gray < 60, 0, gray)
            
        if i % 10 == 0:
                
            gray = cv2.fastNlMeansDenoising(gray,7,7,7)
    #gray = (255-gray)
    return(gray)

def new_preprocessing(img):
        
    if len(img.shape) < 3:
        gray = img
    else:
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
    gray = cv2.bilateralFilter(gray,9,9,9)  
    gray = cv2.fastNlMeansDenoising(gray,7,7,7)
    #gray = cv2.medianBlur(gray, 15)
    #gray = (255-gray)
    gray = normalizeImage(gray)
    

    return(gray)

def sharpening(img):
    #laplacian of gaussian:
    #variables for substracion
    amount = 1
    #variables for laplace
    ddepth = cv2.CV_16S
    kernel_size = 3
    #gaussian
    blur = cv2.GaussianBlur(img, (3,3), 1)
    #laplacian
    laplace = cv2.Laplacian(blur, ddepth, ksize = kernel_size)
    conv = cv2.convertScaleAbs(laplace)
    #substraction
    sharp = float(amount +1) * img - float(amount) * conv
    sharp = sharp.round().astype(np.uint8)
    #sharp = normalizeImage(sharp)
    
    #cv2.imshow("laplace", conv)
    #cv2.waitKey()
    
    return(sharp)

#directly from tutorial
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    #"""Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return (sharpened)


def sharp_kernel(img):

    kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, 0]], np.float32)
    #kernel = 1/3 * kernel
    dst = cv2.filter2D(img, -1, kernel)
    dst = normalizeImage(dst)

    return(dst)
    
if __name__ == "__main__":
    main() 
