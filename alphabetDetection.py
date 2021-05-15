from sklearn.datasets import fetch_openml
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

if(not os.environ.get("PYTHONHTTPSVERIFY","") and getattr(ssl,"_create_unverified_context",None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]

print(pd.Series(y).value_counts())

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z  "]

nClasses = len(classes)

xTrain,xTest,yTrain,yTest = train_test_split(X,y,random_state = 9,train_size = 7500,test_size = 2500)
xTrainScale = xTrain/255
xTestScale = xTest/255

clf = LogisticRegression(solver = "saga",multi_class = "multinomial")
clf.fit(xTrainScale,yTrain)

yPrediction = clf.predict(xTestScale)

accuracy = accuracy_score(yTest,yPrediction)
print(accuracy)

cap = cv2.VideoCapture(0)

while True:
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape
        upperLeft = (int(width/2 - 56),int(height/2 - 56))
        bottomRight = (int(width/2 + 56),int(height/2 + 56))
        cv2.rectangle(gray,upperLeft,bottomRight,(0,255,0),2)
        roi = gray[upperLeft[1]:bottomRight[1],upperLeft[0]:bottomRight[0]]
        imagepil = Image.fromarray(roi)
        imagebw = imagepil.convert("L")
        image_bw_resized = imagebw.resize((28,28),Image.ANTIALIAS)
        imagebwresizedinverted = PIL.ImageOps.invert(image_bw_resized)
        pixelFilter = 20
        minpixel = np.percentile(imagebwresizedinverted,pixelFilter)
        imagebwresizedinvertedscaled = np.CLIP(imagebwresizedinverted-minpixel,0,255)
        maxpixel = np.max(imagebwresizedinverted)
        imagebwresizedinvertedscaled = np.asarray(imagebwresizedinvertedscaled)/maxpixel
        testSample = np.array(imagebwresizedinvertedscaled).reshape(1,784)
        testPrediction = clf.predict(testSample)
        print("Predicted Class is" +testPrediction)
        cv2.imshow("frame",gray)
        if(cv2.waitKey(1) & 0xFF == ord("q")):
            break
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()

