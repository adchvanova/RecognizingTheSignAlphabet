import cv2 # библиотека для решения задач компьютерного зрения
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
cap=cv2.VideoCapture(0)
detector= HandDetector(maxHands=1) # так как захват только для одной руки
classifier=Classifier("Model/keras_model.h5","Model/labels.txt")
offset=20 #для того чтобы не вплотную к последней точке было
imgSize = 300 
folder="Data/letter1"
counter=0
labels=["А","Б","В","Г","Д","Е","Ж","З","И","К","Л","М","Н","О","П","Р","С","Т","У","Ф","Ц","Ш","Ъ","Ы","Ю","Я"]
while True:
    success,img=cap.read()
    imgOutput=img.copy()
    hands,img = detector.findHands(img)
    if hands:
        hand=hands[0]# так как мы распознаем только 1 руку
        x,y,w,h=hand['bbox']
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255 # возвращает новый массив заданной формы и типа с единицами 3 видимо отвечает за цвет?? значения даем для правильного цвета integer of 8 bits
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]# стартовая высота:конечная высота,стартовая ширина:конечная ширина
        imgCropShape=imgCrop.shape

        aspectRatio=h/w #находим соотношение сторон
        if aspectRatio>1:
            temp=imgSize/h
            wCalcutation=math.ceil(temp*w)
            if wCalcutation != None:
                imgResize=cv2.resize(imgCrop,(wCalcutation,imgSize))
                imgResizeShape=imgResize.shape
            if imgResize.shape[0] > 0 and imgResize.shape[1] > 0:
                wGap=math.ceil((imgSize-wCalcutation)/2)
                imgWhite[0:imgResizeShape[0], wGap:wCalcutation+wGap] = imgResize  # .shape(a)[исходник] Возвращает форму массива
                preduction,index=classifier.getPrediction(imgWhite,draw=False)
            #print(preduction,index)

        else:
            temp = imgSize / w
            hCalcutation = math.ceil(temp * h)
            if hCalcutation !=None:
                imgResize = cv2.resize(imgCrop, (imgSize, hCalcutation))
                imgResizeShape = imgResize.shape
            if imgResize.shape[0] > 0 and imgResize.shape[1] > 0:
                hGap = math.ceil((imgSize - hCalcutation) / 2)
                imgWhite[hGap:hCalcutation+hGap,0:imgResizeShape[1]] = imgResize  # .shape(a)[исходник] Возвращает форму массива
                preduction, index = classifier.getPrediction(imgWhite,draw=False)

        cv2.putText(imgOutput, labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)
        cv2.imshow("ImageCrop",imgCrop)
       # cv2.imshow("ImageWhite",imgWhite)
    cv2.imshow("Image",imgOutput) #работа с вебкой
    cv2.waitKey(1)#задержка
