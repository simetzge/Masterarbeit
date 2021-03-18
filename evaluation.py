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
    
    for dicts in ocrlist:
        for entry in evaluationdict:
            if entry["Image"].lower() == dicts["image"].lower(): 
                ground = entry["Content"].replace(" ","")
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
                    ratiolist.append(rtxt)
                    ratiolist.append(rbox)
    print("die durchschnittliche Präzision beträgt ")
    print(np.mean(ratiolist))
    return (ocrlist)
            
    for row in evaluationdict:
        print (row)    
    
    for row in (ocrlist):
        print (row)
        
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