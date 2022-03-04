# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:02:26 2021

@author: Simon
"""

import PySimpleGUI as sg      
import main as run
import basics

def main():
    
    basics.settingsFlag()

    mainLayout = [[sg.Button('Settings')],
                  [sg.Text('Log')],
                  [sg.Output(size=(60,15))], 
                  [sg.Button('Run'),sg.Push(), sg.Exit()]
                  ]

    mainWindow = sg.Window('RectDetect', mainLayout)

    
    while True:                             # The Event Loop
        event, values = mainWindow.read()

        if event == "Settings":
            if basics.SETTINGS_LOADED == False:
                basics.loadSettings()
            settingsGUI()
        if event == "Run":
            run.main()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break      
    mainWindow.close()
    
def settingsGUI():
    
    dataTypes = [".jpg", ".png"]
    settingsLayout = [[sg.Text('General Settings', font = (15))],
                      [sg.FolderBrowse('Working Directory', key= "foldin"), sg.InputText(default_text = basics.PATH, key = "path", change_submits = True)],
                      [sg.Checkbox('Use Absolute Path',key=("use_absolute_path"), default = basics.USE_ABSOLUTE_PATH)],
                      [sg.Text('Input Data Type'),sg.Push(), sg.DropDown(dataTypes, key = "input_data_type")],
                      [sg.Text('Rectangle Detection', font = (15))],
                      [sg.Checkbox('Use Template', default = basics.USE_TEMPLATE, size = (20,1), key = "use_template"),sg.Push(), sg.Checkbox('Use Simple Crop', default = basics.SIMPLE_CROP, size = (20,1), key = "use_simple_crop")],
                      [sg.Checkbox('Modify Threshold', default = basics.MODIFY_THRESHOLD, size = (20,1), key = "mod_thresh"),sg.Push(), sg.Checkbox('Use Conturs Based Cut', default = basics.CONT_BASED_CUT, size = (20,1), key = "cont_cut")],
                      [sg.Text("Image Target Size"),sg.Push(), sg.InputText(default_text = basics.IMG_TARGET_SIZE, size = (6,6), key = "img_target_size")],
                      [sg.Text("Minimal Rectangle Size"),sg.Push(), sg.InputText(default_text = basics.MIN_RECT_AREA, size = (6,6), key = "min_rect_area")],
                      [sg.Text("Thresholds For Modify Threshold (Min/Max)"),sg.Push(), sg.InputText(default_text = basics.THRESHOLD_MIN, key = "thresh_min", size = (3,3)), sg.InputText(default_text = basics.THRESHOLD_MAX, key = "thresh_max", size = (3,3))],
                      [sg.Text('OCR', font = (15))],
                      [sg.Checkbox('Use OCR', default = basics.OCR, key = "OCR")],
                      [sg.FileBrowse("Tesseract Path"), sg.InputText(default_text = basics.TESS_PATH, key = "ocrPath", change_submits = True)],
                      [sg.Text("Tesseract Parameters"),sg.Push(), sg.InputText(default_text = basics.OCR_CONFIG, key = "ocr_conf")],
                      [sg.Checkbox('Invert Image', default = basics.INVERT_IMAGE, key = "invert_img")],
                      [sg.Text('Evaluation', font = (15))],
                      [sg.Checkbox('Evaluate OCR', default = basics.EVALUATE, key = "evaluate")],
                      [sg.FileBrowse("Evaluation List"), sg.InputText(default_text = basics.EVALUATION_LIST, key = "evalList", change_submits = True)],
                      [sg.Checkbox('Optimum', default = basics.OPTIMUM, key = "optimum"),sg.Push(),sg.Checkbox('F-Score', default = basics.FSCORE, key = "fscore"),sg.Push(),sg.Checkbox('More Measurements', default = basics.ALL_MEASURES, key = "all_measures")],
                      [sg.Push()],
                      [sg.Button("Save And Quit"), sg.Button("Discard And Quit")]
                      ]
    settingsWindow = sg.Window("Settings", settingsLayout)
    while True:
        event, values = settingsWindow.read()
        #if values["foldin"] != "":
         #   if basics.SETTINGS_LOADED == False:
          #      basics.loadSettings()
           # useAbsolutePath = True
            #absolutePath = values["foldin"]
            #basics.saveSettings()
            
        if event == sg.WIN_CLOSED or event == 'Discard And Quit':
            break
        
        if event == "Save And Quit":
            #general
            basics.ABSOLUTE_PATH = values["path"]
            basics.USE_ABSOLUTE_PATH = values["use_absolute_path"]
            basics.INPUT_FORMAT = values["input_data_type"]
            #rect detect
            basics.USE_TEMPLATE = values["use_template"]
            basics.SIMPLE_CROP = values["use_simple_crop"]
            basics.MODIFY_THRESHOLD = values["mod_thresh"]
            basics.CONT_BASED_CUT = values["cont_cut"]
            basics.IMG_TARGET_SIZE = values["img_target_size"]
            basics.MIN_RECT_AREA = values["min_rect_area"]
            basics.THRESHOLD_MIN = values["thresh_min"]
            basics.THRESHOLD_MAX = values["thresh_max"]
            #ocr
            basics.OCR = values["OCR"]
            basics.TESS_PATH = values["ocrPath"]
            basics.OCR_CONFIG = values["ocr_conf"]
            basics.INVERT_IMAGE = values["invert_img"]
            #evaluation
            basics.EVALUATE = values["evaluate"]
            basics.EVALUATION_LIST = values["evalList"]
            basics.OPTIMUM = values["optimum"]
            basics.FSCORE = values["fscore"]
            basics.ALL_MEASURES = values["all_measures"]
            #basics.ABSOLUTE_PATH = 
            basics.saveSettings()
            
            break
    
    settingsWindow.close()
        
        
#####################################################################################################################################################
#
# call mainGUI
#
#####################################################################################################################################################            

if __name__ == "__main__":
    main() 
    