import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

#Alarm Sound
mixer.init()
sound = mixer.Sound('alarm.wav')

#Defining the haar cascade classifiers from haar cascade files:
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

label = ['Open','Close']

#Loading trained cnn model:
model = load_model('models/cnnCat2.h5')

path = os.getcwd()     #Get working directory

#Get Camera Input
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_DUPLEX

count = 0    #No of faces detected
score = 0    #Increase score if eyes shut
thicc = 2
left_predict = [99]
right_predict = [99]

while(True):
    ret, frame = cap.read()
    #frame = cv2.imread('TestCollage.PNG')                #Capturing frames from the camera input
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          #Converting frames to grayscale cause OpenCV requires grayscale images
    faces = face.detectMultiScale(gray, minNeighbors= 5,scaleFactor=1.1,minSize=(25,25))   #Detecting faces from gray images using the face classifier

    #Detecting Left and Right Eye, will return x,y coordinates and height and width
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    #Creating box around open/closed and score:
    cv2.rectangle(frame, (0,height-50), (275,height), (0,0,0), thickness=cv2.FILLED)

    #Creating box around face:
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)

    #For predicting right eye status using CNN Classifier:
    for (x,y,w,h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]       #Adjust Dimensions
        count = count + 1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)    #Converting to Grayscale
        r_eye = cv2.resize(r_eye,(24,24))                 #Resize image to @4 * 24 pixels because our model is trained on 24*24
        r_eye = r_eye/255                         #Normalize the data for better convergence, all values will be  between 0 and 1
        r_eye = r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye, axis=0)

        right_predict=model.predict_classes(r_eye)           #Predicting eye is closed or open
        if(right_predict[0]==0):
            label = 'Close'

        elif(right_predict[0]==1):
            label = 'Open'

        break

    #For predicting right eye status using CNN Classifier:
    for (x,y,w,h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]       #Adjust Dimensions
        count = count + 1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)    #Converting to Grayscale
        l_eye = cv2.resize(l_eye,(24,24))                 #Resize image to 24 * 24 pixels because our model is trained on 24*24
        l_eye = l_eye/255                          #Normalize the data for better convergence, all values will be  between 0 and 1
        l_eye = l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye, axis=0)

        left_predict=model.predict_classes(l_eye)           #Predicting eye is closed or open
        if(left_predict[0]==0):
            label = 'Close'

        elif(left_predict[0]==1):
            label = 'Open'

        break

    #Combining for Both Eyes
    if(right_predict[0]==0 and left_predict[0]==0):
        score = score + 1                           #Increasing score if both eyes are closed
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)  #Print Closed on the Frame

    else:
        score = score-1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1,cv2.LINE_AA)  # Print Open on the Frame

    if (score < 0):
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (130, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if (score > 10):                    #Setting Threshold to raise alarm
        cv2.imwrite(os.path.join(path, '/Images/image.jpeg'),frame)
        try:
            sound.play()

        except:
            pass

        if(thicc < 16):
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if(thicc < 2):
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



