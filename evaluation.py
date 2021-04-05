# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:51:09 2021

@author: Simon
"""

from basics import *
from flags import *
import difflib


def evaluate(evaluationdict, ocrlist):
    
    ratiolist = []
    #compare dicts with ocr and evaluationlist
    for dicts in ocrlist:
        for entry in evaluationdict:
            #entries are the same if lowercase of both names is a match
            if entry["Image"].lower() == dicts["image"].lower(): 
                #ignore spaces
                ground = entry["Content"].replace(" ","")
                #get ratio for comparison, add to dict
                for i in range(len(dicts)-1):                    
                    key = "rectangle " + str(i)
                    box = dicts[key]["boximage"].replace(" ", "")
                    txt = dicts[key]["textimage"].replace(" ", "")
                    sbox = difflib.SequenceMatcher(None, ground,box)
                    stxt = difflib.SequenceMatcher(None, ground,txt)
                    rbox = round(sbox.ratio(),2)
                    rtxt = round(stxt.ratio(),2)
                    dicts[key]["textratio"] = rtxt
                    dicts[key]["boxratio"] = rbox
    return (ocrlist)
            
        
def comparison(ocrlist):
    for dicts in ocrlist:
        for i in range(len(dicts)-1):
            key = "rectangle " + str(i)
            box = dicts[key]["boximage"].replace(" ", "")
            txt = dicts[key]["textimage"].replace(" ", "")
            if  box == txt:
                dicts[key]["comparison"] = "same"
            else:
                dicts[key]["comparison"] = "different"
    return(ocrlist)


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
    
    #get time and date for output name
    if name =="output":
        now = datetime.now()
        name = now.strftime("%Y_%m_%d_%H_%M")
    
    #open output file
    with open(path + '\\' + folder + '\\' + name + '.csv', 'w') as file:
        #write column names
        writeHeader(file)
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
        avg, txtavg, boxavg = getAverage(outputlist)
        writeFooter(file, avg, txtavg, boxavg)
    
def writeHeader(file):
    file.write("parameters:\n")
    file.write("IMG_TARGET_SIZE," + str(IMG_TARGET_SIZE) + "\n")
    file.write("THRESHOLD_MIN," + str(THRESHOLD_MIN) + "\n")
    file.write("THRESHOLD_MAX," + str(THRESHOLD_MAX) + "\n")
    file.write("CUT_THRESH," + str(CUT_THRESH) + "\n")
    file.write("USE_ABSOLUTE_PATH," + str(USE_ABSOLUTE_PATH) + "\n")
    file.write("ABSOLUTE_PATH," + str(ABSOLUTE_PATH) + "\n")
    file.write("MODIFY_THRESHOLD," + str(MODIFY_THRESHOLD) + "\n")
    file.write("USE_TEMPLATE," + str(USE_TEMPLATE) + "\n")
    file.write("SIMPLE_CROP," + str(SIMPLE_CROP) + "\n")
    file.write("USE_CNN," + str(USE_CNN) + "\n")
    file.write("EVALUATE," + str(EVALUATE) + "\n")
    file.write("CHECK_PICTURE," + str(CHECK_PICTURE) + "\n")
    file.write("\n")

def writeFooter(file, overallaverage, txtaverage, boxaverage):
    file.write("\n")
    file.write("average," + str(overallaverage) + ",")
    file.write("textaverage," + str(txtaverage) + ",")
    file.write("boxaverage," + str(boxaverage) + ",")
    
def getAverage(dictlist):
    txtratio = []
    boxratio = []
    for dicts in dictlist:
        for i in range(len(dicts)-1):
            rect = "rectangle " + str(i)
            for key in dicts[rect]:
                if key == "textratio":
                    txtratio.append(dicts[rect]["textratio"])
                if key == "boxratio":
                    boxratio.append(dicts[rect]["boxratio"])
            
            txtaverage = round(np.mean(txtratio),2)
            boxaverage = round(np.mean(boxratio),2)
            overallaverage = round(np.mean(txtratio + boxratio),2)
    return(overallaverage, txtaverage, boxaverage)
            
