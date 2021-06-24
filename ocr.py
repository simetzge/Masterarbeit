# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:32:32 2021

@author: Simon Metzger

licensend under Attribution-NonCommercial-ShareAlike 3.0 Germany

CC BY-NC-SA 3.0 DE
"""
import pytesseract
import cv2
import numpy as np

from basics import *
from flags import *

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
            t.append(text)
            b.append(boxtext)
            timgs.append(textimg)
            bimgs.append(boximg)
        
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
        output('text', textimg, fileName, str(j))
        output('box', boximg, fileName, str(j))
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
    
    if CONT_BASED_CUT:
        preimg = np.where(preimg < 63, 127, preimg)
    
    preimg = normalizeImage(preimg)
    
    if INVERT_IMAGE:
        preimg = (255-preimg)
    
    text, rotate = image_to_text(preimg)
    if rotate == True:
        
       preimg = cv2.rotate(preimg, cv2.cv2.ROTATE_180)
    
    if mode == 'image_to_box':
        boximg, boxtext = image_to_box(preimg)
        
    #convert to colored img for output
    textimg = cv2.cvtColor(preimg, cv2.COLOR_GRAY2BGR)

    #write text on image
    cv2.putText(textimg, text, (15, 600),cv2.FONT_HERSHEY_SIMPLEX, 3.8, (0, 230, 0), 3)
    if mode == 'image_to_box':
        return(boximg, boxtext)
    if mode == 'image_to_text':
        return(textimg, text)


#####################################################################################################################################################
#
# OCR with pytesseract
#
#####################################################################################################################################################

def image_to_text(img):
    
    #call for pytesseract
    pytesseract.pytesseract.tesseract_cmd=TESS_PATH
    #set default value for rotate flag
    rotate = False
    
    #try to read
    texta = pytesseract.image_to_string(img, config=OCR_CONFIG)
    #rotate and try again
    img = cv2.rotate(img, cv2.cv2.ROTATE_180)
    textb = pytesseract.image_to_string(img, config=OCR_CONFIG)
    
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
    return(text, rotate)

def textsplit(text):
    #rearrange text to avoid crash
    arr = text.split('\n')[0:-1]
    text = ' '.join(arr)
    return(text)

    
#####################################################################################################################################################
#
# OCR with pytesseract and image to boxes
#
#####################################################################################################################################################

def image_to_box(img):
    
    #call for pytesseract
    pytesseract.pytesseract.tesseract_cmd=TESS_PATH
    #set default value for rotate flag
    rotate = False
    
    preimg = preprocessing(img)
    #get characters with bounding boxes
    boxes = pytesseract.image_to_boxes(preimg, config=OCR_CONFIG)
    
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
    #convert to colored img for output
    boximg = cv2.cvtColor(preimg, cv2.COLOR_GRAY2BGR)
    
    #draw every character with its bounding box
    for rows in newstring:
        if rows[0] == "~":
            newstring.remove(rows)
            continue
        x, y, w, h = (int(rows[1]), int(rows[2]), int(rows[3]), int(rows[4]))
       
        #y has to be inverted to be compatible with cv2 functions
        cv2.rectangle(boximg, (x, boximg.shape[0] - y), (w, boximg.shape[0] - h), (0, 255, 0), 2)
        cv2.putText(boximg, rows[0], (int(x + ((w-x) / 2)), int(boximg.shape[0] - y + (y-h)/2)),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        textlist.append(str(rows[0]))    
    text = "".join(textlist)
   
    return(boximg, text)
        

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

    gray = cv2.GaussianBlur(gray, (9,9), 50)#############

                
    gray = cv2.fastNlMeansDenoising(gray,9,9,50)#########

    gray = normalizeImage(gray)

    return(gray)

