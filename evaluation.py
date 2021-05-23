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
                dicts["bestratio"] = None
                #get ratio for comparison, add to dict
                for i in range(len(dicts)-3):
                    key = "rectangle " + str(i)
                    box = dicts[key]["boximage"].replace(" ", "")
                    txt = dicts[key]["textimage"].replace(" ", "")
                    best = dicts["bestguess"].replace(" ", "")
                    
                    getMeasures(ground, txt)                   

                    sbox = difflib.SequenceMatcher(None, ground,box)
                    stxt = difflib.SequenceMatcher(None, ground,txt)
                    rbox = round(sbox.ratio(),2)
                    rtxt = round(stxt.ratio(),2)
                    if box == best:
                        dicts["bestratio"] = rbox
                    if txt == best:
                        dicts["bestratio"] = rtxt
                    dicts[key]["textratio"] = rtxt
                    dicts[key]["boxratio"] = rbox
                
    return (ocrlist)
            
        
def comparison(ocrlist):
    for dicts in ocrlist:
        for i in range(len(dicts)-2):
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
        file.write("image,rectangle,textimage,boximage,comparison,textratio,boxratio,bestguess,bestratio\n")
        #iterate through dictionaries in list and through subdictionaries
        for dicts in outputlist:
            for i in range(len(dicts)-3):
                #write values in csv format
                file.write(dicts["image"] + ",")
                rect = "rectangle " + str(i)
                for key in dicts[rect]:
                    file.write(str(dicts[rect][key]) + ",")
                #file.write(dicts[key]["rectangle"] + ",")
                #file.write(dicts[key]["textimage"] + ",")
                #file.write(dicts[key]["boximage"] + ",")
                #file.write(dicts[key]["comparison"])
                if dicts["bestguess"] == dicts[rect]["boximage"] or dicts["bestguess"] == dicts[rect]["textimage"]:
                    file.write(dicts["bestguess"] + ",")
                    file.write(str(dicts["bestratio"]))
                file.write("\n")
        avg, txtavg, boxavg,bestavg = getAverage(outputlist)
        writeFooter(file, avg, txtavg, boxavg,bestavg)
    
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
    file.write("CONT_BASED_CUT," + str(CONT_BASED_CUT) + "\n")
    file.write("OCR," + str(OCR) + "\n")
    file.write("USE_CNN," + str(USE_CNN) + "\n")
    file.write("EVALUATE," + str(EVALUATE) + "\n")
    file.write("CHECK_PICTURE," + str(CHECK_PICTURE) + "\n")
    file.write("\n")

def writeFooter(file, overallaverage, txtaverage, boxaverage,bestaverage):
    file.write("\n")
    file.write("average," + str(overallaverage) + ",")
    file.write("textaverage," + str(txtaverage) + ",")
    file.write("boxaverage," + str(boxaverage) + ",")
    file.write("bestaverage," + str(bestaverage) + ",")
    
def getAverage(dictlist):
    txtratio = []
    boxratio = []
    bestguessratio = []
    for dicts in dictlist:
        for i in range(len(dicts)-3):
            rect = "rectangle " + str(i)
            for key in dicts[rect]:
                if key == "textratio":
                    txtratio.append(dicts[rect]["textratio"])
                if key == "boxratio":
                    boxratio.append(dicts[rect]["boxratio"])
        bestguessratio.append(dicts["bestratio"])
    txtaverage = round(np.mean(txtratio),2)
    boxaverage = round(np.mean(boxratio),2)
    overallaverage = round(np.mean(txtratio + boxratio),2)
    bestaverage = round(np.mean(bestguessratio),2)
    return(overallaverage, txtaverage, boxaverage,bestaverage)

##################################################################################################################################################### 
#
# input: 2 strings (ground truth and detected text)
# ouput: true positives (tp), false positives (fp), false negatives (fn)
# purpose: get tp, fp, fn to calculate precision, recall and f-score
#
#####################################################################################################################################################
            
def getMeasures(ground, text):
    grounddict ={}
    textdict ={}    
    tp = 0
    fp = 0
    fn = 0
    #count letters in ground, save in dict
    for letter in ground:
        if letter in grounddict: 
            grounddict[letter] = grounddict[letter]+1
        else:
            grounddict[letter] = 1
            textdict[letter] = 0
            
    #count letters in text, save in dict         
    for letter in text:
        if letter in textdict: 
            textdict[letter] = textdict[letter]+1
        else:
            textdict[letter] = 1
        if letter not in grounddict:
            grounddict[letter] = 0
    #print("ground\n")
    #print(grounddict)
    #print("text\n")
    #print(textdict)
    if len(grounddict) > 0 and len(textdict) > 0:
        for key in grounddict:
            g = grounddict[key]
            t = textdict[key]
            if g <= t:
                tp = tp + g
            if t > g:
                fp = fp + (t - g)
            if g > t:
                tp = tp + t
                fn = fn + (g -t)
    #print("tp: " + str(tp))
    #print("fp: " + str(fp))
    #print("fn: " + str(fn))            
    return(tp,fp,fn)
