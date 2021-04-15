# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:58:04 2021

@author: Simon
"""

import os
import re
import cv2
import numpy as np
import csv
import itertools
from datetime import datetime

from flags import *

#####################################################################################################################################################
#    
# function for searching all files with the matching extension in the input directory
# will return the paths and names of all files found
#
#####################################################################################################################################################

def searchFiles(extension,folder):
        
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
        if folder in dirs:
            print(folder + ' gefunden')
            
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
            print(folder + ' fehlt')
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

def scaleImage(img, size = IMG_TARGET_SIZE):
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
    if USE_ABSOLUTE_PATH == True:
        path = ABSOLUTE_PATH
    else:
        path = os.getcwd()
        
    #list all files in path
    dirs = os.listdir(path)
        
    #mnake empty array for input files
    files = []
    names = []
    csvdict = []
        
    #if input dir found
    if folder in dirs:
        print(folder + ' gefunden')
        #list all files in input dir
        content = path + '\\' + folder + '\\' + inputFile
        with open(content , newline='') as csvfile:
            #csvlist = list(csv.reader(csvfile, delimiter = ',', quotechar = '"'))
            reader = csv.DictReader(csvfile)
            for line in reader:
                csvdict.append(line)
            return(csvdict)
    else:
        print(folder + ' nicht gefunden')
        
def show(img, name = "show"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL) 
    img =cv2.resize(img, (1000,750))    
    cv2.imshow("show", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
