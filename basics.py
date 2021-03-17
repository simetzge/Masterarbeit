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

def scaleImage(img):
        scale = IMG_TARGET_SIZE / np.max(img.shape)
        return (cv2.resize(img, (0,0), fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC))
    
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
    
#####################################################################################################################################################
#      
# evaluation output as csv
#
#####################################################################################################################################################

def csvOutput_csv(csvFile, folder = "evaluation", name = "output"):
    
    #check wich patch should be used
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
    fields = ["image", "rectangle", "textimage", "boximage"]
    with open(path + '\\' + folder + '\\' + name + '.csv', 'w') as f:
        
        for dicts in csvFile:
            writer = csv.DictWriter(f, fields)
            for key, val in sorted(dicts.items()):
                row = {'image': key}
                row.update(val)
                writer.writerow(row)
            #for key in dicts:
                #writer.writerow({field: dicts[key].get(field) or key for field in fields})
    #with open(name + ".csv", 'w', newline='') as csvfile:
        
def csvOutput(outputlist, folder = "evaluation", name = "output"):
    
    #check wich patch should be used    
    if USE_ABSOLUTE_PATH == True:
        path = ABSOLUTE_PATH
    else:
        path = os.getcwd()
    #list all files in path
    dirs = os.listdir(path)
    #if folder doesn't exist create it
    if folder in dirs:
        print(folder + '-Ordner vorhanden')
    else:
        os.makedirs(path + '\\' + folder)

    #open output file
    with open(path + '\\' + folder + '\\' + name + '.csv', 'w') as file:
        #write header
        file.write("image,rectangle,textimage,boximage,comparison,textratio,boxratio\n")
        #iterate through dictionaries in list and through subdictionaries
        for dicts in outputlist:
            for i in range(len(dicts)-1):
                #write values in csv format
                file.write(dicts["image"] + ",")
                rect = "rectangle " + str(i)
                for key in dicts[rect]:
                    file.write(str(dicts[rect][key]) + ",")
                #file.write(dicts[key]["rectangle"] + ",")
                #file.write(dicts[key]["textimage"] + ",")
                #file.write(dicts[key]["boximage"] + ",")
                #file.write(dicts[key]["comparison"])
                file.write("\n")
            