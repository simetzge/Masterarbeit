# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:02:26 2021

@author: Simon
"""

import PySimpleGUI as sg      
import main
import basics

def main():
    
    basics.settingsFlag()

    mainLayout = [[sg.FolderBrowse('Working Directory', key= "foldin"),sg.Button('Settings')],
                  [sg.Text('Log')],
                  [sg.Output(size=(60,15))], 
                  [sg.Button('Run'),sg.Push(), sg.Exit()],
                  [sg.Button('Test')]]

    mainWindow = sg.Window('RectDetect', mainLayout)

    
    while True:                             # The Event Loop
        event, values = mainWindow.read()

        if event == "Settings":
            if basics.SETTINGS_LOADED == False:
                basics.loadSettings()
            settingsGUI()
        if event == "Run":
            main.main()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break      
        if event == "Test":
            #test = basics.Setting("general settings", "USE_OCR")
        
            #if  test == True:
            #    print("yes")
            #else:
            #   print(test)
            if basics.SETTINGS_LOADED != True:
                basics.loadSettings()
            else:
                print("Settings already loaded")
            #print(basics.PATH)
            print(values["foldin"])
            #absolutePath()
            #print(ABSOLUTE_PATH)
    mainWindow.close()
    
def settingsGUI():
    
    dataTypes = [".jpg", ".png"]
    settingsLayout = [[sg.Text('General Settings', font = (15))],
                      [sg.FolderBrowse('Working Directory', key= "foldin")],
                      [sg.Checkbox('Use Absolute Path', default = basics.USE_ABSOLUTE_PATH)],
                      [sg.Text('Input Data Type'),sg.Push(), sg.DropDown(dataTypes)],
                      [sg.Text('Rectangle Detection', font = (15))],
                      [sg.Checkbox('Use Template', default = basics.USE_TEMPLATE, size = (20,1)),sg.Push(), sg.Checkbox('Use Simple Crop', default = basics.SIMPLE_CROP, size = (20,1))],
                      [sg.Checkbox('Modify Threshold', default = basics.MODIFY_THRESHOLD, size = (20,1)),sg.Push(), sg.Checkbox('Use Conturs Based Cut', default = basics.CONT_BASED_CUT, size = (20,1))],
                      [sg.Text("Image Target Size"),sg.Push(), sg.InputText(default_text = basics.IMG_TARGET_SIZE, size = (6,6))],
                      [sg.Text("Minimal Rectangle Size"),sg.Push(), sg.InputText(default_text = basics.MIN_RECT_AREA, size = (6,6))],
                      [sg.Text("Thresholds For Modify Threshold (Min/Max"),sg.Push(), sg.InputText(default_text = basics.THRESHOLD_MIN, size = (3,3)), sg.InputText(default_text = basics.THRESHOLD_MAX, size = (3,3))],
                      [sg.Text('OCR', font = (15))]
                      ]
    settingsWindow = sg.Window("Settings", settingsLayout)
    while True:
        event, values = settingsWindow.read()
        if values["foldin"] != "":
            if basics.SETTINGS_LOADED == False:
                basics.loadSettings()
            useAbsolutePath = True
            absolutePath = values["foldin"]
            #basics.saveSettings()
            
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
    
    settingsWindow.close()
        
        
#####################################################################################################################################################
#
# call mainGUI
#
#####################################################################################################################################################            

if __name__ == "__main__":
    main() 
    