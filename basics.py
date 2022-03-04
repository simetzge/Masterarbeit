# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:58:04 2021

@author: Simon Metzger

licensend under Attribution-NonCommercial-ShareAlike 3.0 Germany

CC BY-NC-SA 3.0 DE
"""

import os
import re
import cv2
import numpy as np
import csv
import itertools
from datetime import datetime

import configparser
#from flags import *


#####################################################################################################################################################
#    
# function for searching all files with the matching extension in the input directory
# will return the paths and names of all files found
#
#####################################################################################################################################################

def searchFiles(extension,folder):
        
        #get skript path

        path = PATH
        
        #list all files in path
        dirs = os.listdir(path)
        
        #mnake empty array for input files
        files = []
        names = []
        
        #if input dir found
        if folder in dirs:
            print(folder + ' found!')
            
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
            print(folder + ' not found!')
        #return all found files
        return(files, names)

#####################################################################################################################################################
#
# writes given images with given names and an extension for the modification, e.g. 'blurred'
# will create outputfolder in program path
#
#####################################################################################################################################################

def output(folder, img, name, mod=''):
        path = PATH
        #list all files in path
        dirs = os.listdir(path)
        if folder in dirs:
            print(folder + '-folder found!')
        else:
            os.makedirs(path + '\\' + folder)
            print(folder + '-folder created!')
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

def scaleImage(img, size = 0):
        if size == 0:
            size = IMG_TARGET_SIZE
        scale = size / np.max(img.shape)
        if SIMPLE_CROP:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA
        return (cv2.resize(img, (0,0), fx = scale, fy = scale, interpolation = interpolation))
    
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
# open csv files
#
#####################################################################################################################################################
    
def csvInput(inputFile, folder = "evaluation"):
    
    #get skript path
    path = PATH   
    #list all files in path
    dirs = os.listdir(path)
        
    #mnake empty array for input files
    files = []
    names = []
    csvdict = []
        
    #if input dir found
    if folder in dirs:
        print(folder + ' found!')
        #list all files in input dir
        content = path + '\\' + folder + '\\' + inputFile
        with open(content , newline='') as csvfile:
            #csvlist = list(csv.reader(csvfile, delimiter = ',', quotechar = '"'))
            reader = csv.DictReader(csvfile)
            for line in reader:
                csvdict.append(line)
            return(csvdict)
    else:
        print(folder + ' not found!')

def ocrOutput(ocrList, folder = "OCR", name = "output"):
    
    #check wich patch should be used    
    path = PATH
    #list all files in path
    dirs = os.listdir(path)
    #if folder doesn't exist create it
    if folder in dirs:
        print(folder + '-folder found!')
    else:
        os.makedirs(path + '\\' + folder)
    
    #get time and date for output name
    if name =="output":
        now = datetime.now()
        name = now.strftime("%Y_%m_%d_%H_%M")
    
    #open output file
    with open(path + '\\' + folder + '\\' + name + '.csv', 'w') as file:
        #write column names
        file.write("image,textimage,boximage,bestguess\n")
        #iterate through dictionaries in list and through subdictionaries
        for dicts in ocrList:
            printed = False
            for i in range(len(dicts)-2):
                #write values in csv format
                file.write(dicts["image"] + ",")
                rect = "rectangle " + str(i)
                file.write(str(dicts[rect]["textimage"]) + ",")
                file.write(str(dicts[rect]["boximage"]) + ",")
                #file.write(dicts[key]["rectangle"] + ",")
                #file.write(dicts[key]["textimage"] + ",")
                #file.write(dicts[key]["boximage"] + ",")
                #file.write(dicts[key]["comparison"])
                if (dicts["bestguess"] == dicts[rect]["boximage"] or dicts["bestguess"] == dicts[rect]["textimage"]) and printed == False:
                    file.write(dicts["bestguess"])
                    printed = True
                file.write("\n")

#####################################################################################################################################################
#      
# get settings from settings.ini and provide them as global variables
#
#####################################################################################################################################################


def loadSettings():    
    settings = configparser.ConfigParser()
    path = os.getcwd()
    settings.read(path + "\\test.ini")
    
    #general settings
    global USE_ABSOLUTE_PATH
    USE_ABSOLUTE_PATH = settings.getboolean("general settings", "USE_ABSOLUTE_PATH")
    global ABSOLUTE_PATH
    ABSOLUTE_PATH = settings.get("general settings", "ABSOLUTE_PATH")
    global PATH
    if USE_ABSOLUTE_PATH == True:
        PATH = ABSOLUTE_PATH
    else:
        PATH = path
        
    global INPUT_FORMAT
    INPUT_FORMAT = settings.get("general settings", "INPUT_FORMAT")

    global CHECK_PICTURE
    CHECK_PICTURE = settings.get("general settings", "CHECK_PICTURE")
    
    #rect detection
    
    global MODIFY_THRESHOLD
    MODIFY_THRESHOLD = settings.getboolean("rect detection", "MODIFY_THRESHOLD")
    
    global USE_TEMPLATE
    USE_TEMPLATE = settings.getboolean("rect detection", "USE_TEMPLATE")
    
    global SIMPLE_CROP
    SIMPLE_CROP = settings.getboolean("rect detection", "SIMPLE_CROP")

    global CONT_BASED_CUT
    CONT_BASED_CUT = settings.getboolean("rect detection", "CONT_BASED_CUT")

    global IMG_TARGET_SIZE
    IMG_TARGET_SIZE = settings.getint("rect detection", "IMG_TARGET_SIZE")
    
    global MIN_RECT_AREA
    MIN_RECT_AREA = settings.getint("rect detection", "MIN_RECT_AREA")
    
    global THRESHOLD_MIN
    THRESHOLD_MIN = settings.getint("rect detection", "THRESHOLD_MIN")
    
    global THRESHOLD_MAX
    THRESHOLD_MAX = settings.getint("rect detection", "THRESHOLD_MAX")
    
    global CUT_THRESH
    CUT_THRESH = settings.getint("rect detection", "CUT_THRESH")
    
    #OCR
    global OCR
    OCR = settings.getboolean("ocr", "USE_OCR")

    global TESS_PATH
    TESS_PATH = settings.get("ocr", "TESS_PATH")

    global OCR_CONFIG
    OCR_CONFIG = settings.get("ocr", "OCR_CONFIG")

    global INVERT_IMAGE
    INVERT_IMAGE = settings.getboolean("ocr", "INVERT_IMAGE")

    #evaluation
    global EVALUATE
    if OCR == False:
        EVALUATE = False
    else:
        EVALUATE = settings.getboolean("evaluation", "EVALUATE")     
  
    global EVALUATION_LIST
    EVALUATION_LIST = settings.get("evaluation", "EVALUATION_LIST")
        
    global OPTIMUM
    OPTIMUM = settings.getboolean("evaluation", "OPTIMUM")
    
    global FSCORE
    FSCORE = settings.getboolean("evaluation", "FSCORE")
    
    global ALL_MEASURES
    ALL_MEASURES = settings.getboolean("evaluation", "ALL_MEASURES")    

    global SETTINGS_LOADED
    SETTINGS_LOADED = True
    print("Settings loaded")

#####################################################################################################################################################
#      
# save the global settings variables in a settings.ini
#
#####################################################################################################################################################

def saveSettings():
    settings = configparser.ConfigParser(allow_no_value=True)
        
    #general settings
    settings.add_section('general settings')
    settings.set('general settings', '#change input path (default = same path as pyhon files)', None)
    settings.set("general settings", "USE_ABSOLUTE_PATH", str(USE_ABSOLUTE_PATH))
    settings.set("general settings", "ABSOLUTE_PATH", str(ABSOLUTE_PATH))
    settings.set('general settings', '#input data format', None)
    settings.set("general settings", "INPUT_FORMAT", str(INPUT_FORMAT))
    settings.set('general settings', '#only use picture with this name', None)
    settings.set("general settings", "CHECK_PICTURE", str(CHECK_PICTURE))
    
    #rect detection
    settings.add_section('rect detection')    
    settings.set('rect detection', '#search for a template to get aspect ratio', None)
    settings.set("rect detection", "USE_TEMPLATE",str(USE_TEMPLATE))
    settings.set('rect detection', '#use simple crop instead of the more complex hough based crop', None)
    settings.set("rect detection", "SIMPLE_CROP",str(SIMPLE_CROP))
    settings.set('rect detection', '#use iterative instead of adaptive threshold', None)
    settings.set("rect detection", "MODIFY_THRESHOLD",str(MODIFY_THRESHOLD))
    settings.set('rect detection', '#use cut based on contours instead of rects (not recommended)', None)
    settings.set("rect detection", "CONT_BASED_CUT",str(CONT_BASED_CUT))
    settings.set('rect detection', '#size for downscaling', None)
    settings.set("rect detection", "IMG_TARGET_SIZE",str(IMG_TARGET_SIZE))
    settings.set('rect detection', '#min size of rectangle', None)
    settings.set("rect detection", "MIN_RECT_AREA",str(MIN_RECT_AREA))
    settings.set('rect detection', '#thresholds for modifiy threshold', None)
    settings.set("rect detection", "THRESHOLD_MIN",str(THRESHOLD_MIN))
    settings.set("rect detection", "THRESHOLD_MAX",str(THRESHOLD_MAX))
    settings.set('rect detection', '#threshold for hough based crop', None)
    settings.set("rect detection", "CUT_THRESH",str(CUT_THRESH))
    
    #OCR
    settings.add_section('ocr')
    settings.set('ocr', '#use OCR', None)
    settings.set("ocr", "USE_OCR", str(OCR))
    settings.set('ocr', '#tesseract path', None)
    settings.set("ocr", "TESS_PATH",str(TESS_PATH))
    settings.set('ocr', '#tesseract config', None)
    settings.set("ocr", "OCR_CONFIG",str(OCR_CONFIG))
    settings.set('ocr', '#invert image (use this if the background is darker than the text)', None)
    settings.set("ocr", "INVERT_IMAGE",str(INVERT_IMAGE))

    #evaluation
    settings.add_section('evaluation')
    settings.set('evaluation', '#evaluate OCR', None)
    settings.set("evaluation", "EVALUATE",str(EVALUATE))
    settings.set('evaluation', '#name of evaluation list', None)     
    settings.set("evaluation", "EVALUATION_LIST",str(EVALUATION_LIST))
    settings.set('evaluation', '#add optmimum value to evaluation', None)
    settings.set("evaluation", "OPTIMUM",str(OPTIMUM))
    settings.set('evaluation', '#F-Score instead of ratio', None)
    settings.set("evaluation", "FSCORE",str(FSCORE))
    settings.set('evaluation', '#add recall, precision to evaluation', None)
    settings.set("evaluation", "ALL_MEASURES",str(ALL_MEASURES))
    #open file and write settings
    with open('test.ini', 'w') as fp:
        settings.write(fp)
    #reset settings loaded flag
    settingsFlag()
    print("Settings saved.")
    
#####################################################################################################################################################
#      
# set flag if settings need to be loaded (at the beginning and after changes)
#
#####################################################################################################################################################
          
def settingsFlag():
    #flag to set if settings need to be loaded (at the beginning and after changes)
    global SETTINGS_LOADED
    SETTINGS_LOADED = False
        

#####################################################################################################################################################
#      
# show image for debug
#
#####################################################################################################################################################
        
def show(img, name = "show"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL) 
    img =cv2.resize(img, (1000,750))    
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
