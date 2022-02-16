# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:02:26 2021

@author: Simon
"""

import PySimpleGUI as sg      
from main import *
from basics import *

layout = [[sg.FolderBrowse('Working Directory'),sg.Button('Settings')],
          [sg.Text('Log')],
          [sg.Output(size=(60,15))], 
          [sg.Button('Run'),sg.Push(), sg.Exit()],
          [sg.Button('Test')]]

window = sg.Window('RectDetect', layout)    

while True:                             # The Event Loop
    event, values = window.read()
    if event == "Run":
        main()
    if event == sg.WIN_CLOSED or event == 'Exit':
        break      
    if event == "Test":
        test = getSetting("general settings", "USE_OCR")
        if  test == True:
            print("yes")
        else:
            print(test)
window.close()