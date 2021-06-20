
import os
import cv2

path = os.getcwd()
dirs = os.listdir(path)
files = []
names = []
count = []

allcont = []

for folder in dirs:
    if ".py" in folder: continue
    content = os.listdir(path + "\\" + folder)
    if "collection" in folder: continue
    for item in content:
        files.append(path + '\\' + folder + '\\' + item)
        names.append(item)

images = [cv2.imread(file) for file in files]

if "collection" in dirs:
    print("collection" + '-Ordner vorhanden')
else:
    os.makedirs(path + '\\' + "collection")
    
for i, img in enumerate(images):
    count = 0
    outputcont = os.listdir(path + "\\collection")
    while names[i] in outputcont:
        count = count + 1 
        names[i] = names[i][:-5] + str(count) + ".png"
    cv2.imwrite(path + '\\collection\\' + names[i], img)    
        
print("test")


    
