# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:32:32 2021

@author: Simon
"""
import pytesseract
import cv2
import numpy as np
from basics import *

########################
#for debug only
from evaluation import *
from flags import *

def main():
    
    indiOCR(folder = "crop adaptive simple")
    indiOCR(folder = "crop adaptive hough")
    indiOCR(folder = "crop iterative simple")
    indiOCR(folder = "crop iterative hough")
    indiOCR(folder = "hough only")
    indiOCR(folder = "simple only")
    indiOCR(folder = "adaptive only")
    indiOCR(folder = "iterative only")
    #indiOCR(folder = "crop")
    indiOCR(folder = "collection")

def indiOCR(folder = "crop"):
    #main function for direct testing and comparison of OCR methods, for debug only
    filePaths, fileNames = searchFiles(".png", folder)

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
    
    evaluation = csvInput("evaluationlist.csv")
    compared = comparison(ocrlist)
    evaluated = evaluate(evaluation, compared)
    csvOutput(evaluated)


def ocr(cropimgs, fileName):
    imagedict = {"image": fileName}
    bestguess = ""
    for j, crop in enumerate(cropimgs):
        data = []
        timgs = []
        bimgs = []
        t = []
        b = []
        for k in range(3):
            onechannel = crop.copy()
            if not k == 0 :onechannel[:,:,0] = 0
            if not k == 1 :onechannel[:,:,1] = 0
            if not k == 2 :onechannel[:,:,2] = 0
            textimg, text = get_text(onechannel)
            boximg, boxtext = get_text(onechannel, mode = 'image_to_box')
            #data.append(get_text(onechannel, mode = "image_to_data"))
            t.append(text)
            b.append(boxtext)
            timgs.append(textimg)
            bimgs.append(boximg)
        
        #data.append(get_text(onechannel, mode = "image_to_data"))
        textimg, text = get_text(crop)
        boximg, boxtext = get_text(crop, mode = 'image_to_box')
        t.append(text)
        b.append(boxtext)
        timgs.append(textimg)
        bimgs.append(boximg)
        
        #find longest text                
        for n, txt in enumerate(t):
            tx = txt.replace(" ", "")
            tex = text.replace(" ", "")
            if len (tx) > len(tex):
                text = txt
                textimg = timgs[n]
        #find longest boxtext        
        for n, box in enumerate(b):
            bx = box.replace(" ", "")
            boxtex = boxtext.replace(" ", "")
            if len (bx) > len(boxtex):
                boxtext = box
                boximg = bimgs[n]
                
        #delete spaces, check if box or txt are longer than bestguess, replace bestguess in this case
        box = boxtext.replace(" ", "")
        txt = text.replace(" ", "")
        if max(len(box),len(txt)) > len(bestguess):
            if len(box) > len(txt):
                bestguess = boxtext
            else:
                bestguess = text
        #output of rectimage and boximage
        output('rect', textimg, fileName, str(j))
        output('box', boximg, fileName, str(j))
        #output('crop', crop, fileName, str(j))
        #output('rect', crop, fileNames[i], str(j))
        #add rectdict to imagedict and imagedict to ocrlist
        rectdict = {"rectangle": fileName + "_" + str(j), "textimage": text, "boximage": boxtext}
        imagedict["rectangle " + str(j)] =  rectdict
        imagedict["bestguess"] = bestguess
        
    return(imagedict)            
               
            
#####################################################################################################################################################
#
# Calls the ocr subfunctions. Returns processed image with text on it
#
#####################################################################################################################################################     

def get_text(img, mode = 'image_to_text'):
  
    preimg = preprocessing(img)
    
    #preimg = getinnerrect(preimg)
    
    #preimg = new_preprocessing(preimg)
    if CONT_BASED_CUT:
        preimg = np.where(preimg < 63, 127, preimg)
    
    preimg = normalizeImage(preimg)
    
    preimg = (255-preimg)
    
    text, rotate = image_to_text(preimg)
    if rotate == True:
        
       preimg = cv2.rotate(preimg, cv2.cv2.ROTATE_180)
    
    if mode == 'image_to_box':
        boximg, boxtext = image_to_box(preimg)
        
    #write text on image
    
    #convert to colored img for output
    textimg = cv2.cvtColor(preimg, cv2.COLOR_GRAY2BGR)
    
    #color the darkest areas for debug
    #preimg = np.where(preimg < 20, 255, preimg)
    #for columns in preimg:
        #for rows in columns:
            #if rows[0] == 255 and rows[1] == 255 and rows[2] == 255:
                #rows[0] = 0
                #rows[1] = 0
    
    cv2.putText(textimg, text, (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 230, 0), 3)
    if mode == 'image_to_data':
        return(data)
    if mode == 'image_to_box':
        return(boximg, boxtext)
    if mode == 'image_to_text':
        return(textimg, text)

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
    config = ("board -l dic --oem 1 --psm 3")
    #config = ("newboard --oem 1 --psm 7")

    texta = pytesseract.image_to_string(img, config=config)
    #rotate and try again
    img = cv2.rotate(img, cv2.cv2.ROTATE_180)
    textb = pytesseract.image_to_string(img, config=config)
    
    #take the version with more chars detected and send them to textsplit for proper text output, set rotate to True if necessary
    comparea = texta.replace(" ", "")
    comparea = comparea.replace("\n", "")
    compareb = textb.replace(" ", "")
    compareb = compareb.replace("\n", "")
    if len(comparea) >= len(compareb):
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
    text = ' '.join(arr)
    return(text)

#####################################################################################################################################################
#
# OCR with pytesseract and image to data
#
#####################################################################################################################################################

def image_to_data(img):
    
    #call for pytesseract
    pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    #set default value for rotate flag
    rotate = False
    
    #convert to colored img for output
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #try to read
    #config = ('board')
    config = ("newboard --oem 1 --psm 7")
    data = pytesseract.image_to_boxes(img, config=config)
    
    #for box in boxes:
     #   for item in box:
      #      print("item " + item + " item end")
        #letter = box[0]
        #rect = (box[1], box[2]), (box[3], box[4]), box[5]
        #add the found rectangles in green to image
        #img = cv2.drawContours(img, cv2.boxPoints(rect).astype('int32'), -1, (0, 230, 0))
        #cv2.putText(img, letter, (box[1], box[2]),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    from pytesseract import Output
    d = pytesseract.image_to_data(img, config=config, output_type=Output.DICT)
    cv2.imshow("box", img)
    cv2.waitKey()
    
#####################################################################################################################################################
#
# OCR with pytesseract and image to boxes
#
#####################################################################################################################################################

def image_to_box(img):
    
    #call for pytesseract
    pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    #set default value for rotate flag
    rotate = False
    
    #convert to colored img for output
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = preprocessing(img)
    #config = ('board')
    config = ("board -l dic --oem 1 --psm 3")
    #config = ("newboard --oem 1 --psm 7")
    #get characters with bounding boxes
    boxes = pytesseract.image_to_boxes(img, config=config)
    
    #boxes is a list, where every single letter is a seperate entry. The list is casted into a string, the string is split at every space
    string = ''.join(boxes)
    string = string.split()

    j = 0
    newstring = []
    row = []
    rects = []
    textlist = []
    
    #split the resulting list in a 2d list with 6 variables per row
    for i in range(len(string)):
        row.append(string[i])
        if len(row) == 6:
            newstring.append(row)
            row = []
    
    #draw every character with its bounding box
    for rows in newstring:
        if rows[0] == "~":
            newstring.remove(rows)
            continue
        x, y, w, h = (int(rows[1]), int(rows[2]), int(rows[3]), int(rows[4]))
        #angle = int(rows[5])
        #rect = (x, y), (w, h), angle
        
        #y has to be inverted to be compatible with cv2 functions
        cv2.rectangle(img, (x, img.shape[0] - y), (w, img.shape[0] - h), (0, 255, 0), 2)
        cv2.putText(img, rows[0], (int(x + ((w-x) / 2)), int(img.shape[0] - y + (y-h)/2)),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        textlist.append(str(rows[0]))    
        #img = cv2.drawContours(img, [cv2.boxPoints(rect).astype('int32') for rect in rects],-1, (0, 230, 0))
    text = "".join(textlist)
    #cv2.imshow("box", img)
    #cv2.waitKey()    
    return(img, text)
        

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
    #for i in range (10):#chenged from 10 to 1 for evaluation test
    #gray = cv2.GaussianBlur(gray, (9,9), 50)       
    #gray = cv2.medianBlur(gray, 3)
    gray = cv2.GaussianBlur(gray, (9,9), 50)#############
        #blur = cv2.GaussianBlur(img, (3,3), 1)
 
               
        # set everything lower than 50 to 0
        #gray = np.where(gray < 60, 0, gray)            
        #if i % 10 == 0:
                
    gray = cv2.fastNlMeansDenoising(gray,9,9,50)#########
    gray = normalizeImage(gray)
    
    gray = unsharp_mask(gray, threshold = 3)
    #gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -5)
    #gray = laplace(gray)
    #gray = normalizeImage(gray)
    #ret, gray = cv2.threshold(gray, 100, THRESHOLD_MAX, cv2.THRESH_BINARY)  
    #gray = (255-gray)
    #show (gray)
    #mean = np.mean(gray)
    #gray = np.where(gray < (mean/2), mean, gray)
    #gray = normalizeImage(gray)  
    ######### failed atempt with contours etc
    #gray = cannyThreshold(gray)
    #show(gray)
    #contours, hierarchy  = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #contours = [cv2.convexHull(contour) for contour in contours]
    #contAreas = [cv2.contourArea(contour) for contour in contours]   
    #gray = contMask(gray, convex)
    #color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    #newImg = np.zeros(img.shape,np.uint8)
    #newImg[:,:,:] = 2550
    #newImg = cv2.drawContours(mask, contours, -1, (0, 0, 0), 2)
    #for i, contour in enumerate(contours):
        #if contAreas[i] < 10:
            #continue
    #    newImg = cv2.drawContours(newImg, contour, -1, (0, 0, 0), 2)
    #newImg = cv2.drawContours(mask, convex, -1, (0, 0, 0), 2)
    #gray = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)
    #show(gray)
    #gray = (255-gray)
    
    ######### erosion
    #gray = erosion(gray)
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


    #skeleton from opencv doc

def skeleton(img):
        # Step 1: Create an empty skeleton
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)
        
        # Get a Cross Shaped Kernel
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        
        # Repeat steps 2-4
        #while True:
            #Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            #Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
            #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
            # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
            #if cv2.countNonZero(img)==0:
             #   break
        return(skel)
    
def erosion(img):
    

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    return(erosion)    

def cannyThreshold(img):
        max_lowThreshold = 100
        ratio = 3
        kernel_size = 3
        low_threshold = 1
        img_blur = cv2.blur(img, (5,5))
        #detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
        detected_edges = cv2.Canny(img_blur, 10, 30, kernel_size)
        mask = detected_edges != 0
        dst = img * (mask[:,:].astype(img.dtype))
        return (dst)
def contMask(img,contours):
        cropImgs = []
        for contour in contours:
            mask = np.zeros(img.shape,np.uint8)
            cv2.drawContours(mask,[contour],0,(0),-1)
            #show(mask)
            cut = cv2.bitwise_and(mask, img)
            #show(cut)
            cropImgs.append(cut)
        return(cropImgs)  
def laplace(img):
    
    
    ddepth = cv2.CV_16S
    kernel_size = 3
    
    dst = cv2.Laplacian(img, ddepth, ksize=kernel_size)
    return(cv2.convertScaleAbs(dst))

    
if __name__ == "__main__":
    main() 
