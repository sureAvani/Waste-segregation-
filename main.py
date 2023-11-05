import os
import cvzone
from cvzone.ClassificationModule import Classifier
import cv2

cap = cv2.VideoCapture(0)
classifier = Classifier('resources/Model/keras_model.h5', 'resources/Model/labels.txt')

# import all waste images
imgWasteList = []
pathFolderWaste = "resources/waste"
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))


while True:
    _, img = cap.read()
    imgResize = cv2.resize(img, (527, 527))

    imgBackground = cv2.imread('resources/background.png')

    predection = classifier.getPrediction(img)
    print(predection)

    cvzone.overlayPNG(imgBackground, imgWasteList[0], (909, 127))

    imgBackground[315:315+527, 122:122+527] = imgResize
    #Display
    cv2.imshow("Image", img)
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)
