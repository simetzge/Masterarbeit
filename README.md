# Masterarbeit

This repository contains a tool for the detection of boards in archeological excavations.

To get the tool running, a few steps are necessary:

- The tool is started via main.py.
- The tool needs input. Create an input folder in the program path or in any other path and provide this path via the ABSOLUTE_PATH-Flag in flags.py. Default input type is jpg. It can be changed in flags too.
- A local version of tesseract is needed, e.g. the version provided by the [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). The tesseract path should be added in the flags.py.
- The packages numpy, difflib, opencv and pytesseract are mandatory.
- Yolo and Coco weights are not included. The CNN-part won't work without them.
- The folder "example" contains everything needed to run a test:
  - Use the image "testimage" as test image.
  - Copy the "tessdata" folder into you "Tesseract-OCR" folder to use the dictionary and whitelist or change the ocr config in flags.py accordingly.
