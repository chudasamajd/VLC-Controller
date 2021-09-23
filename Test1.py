import cv2
from keras.models import load_model
import keras
import numpy as np
import pandas as pd
from keyboard import Keyboard


model = load_model(r"D:\Python Projects\VLCController\resnetmodel.hdf5")

labels = pd.read_csv(r"D:\Python Projects\VLCController\jester-v1-labels.csv",header=None)

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH,400)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT,400)

buffer = []
cls = []
predicted_value = 0
final_label = ""
i = 1

while(vid.isOpened()):
    flag, frame = vid.read()
    if flag:
        image = cv2.resize(frame,(96,64))
        image = image/255.0
        buffer.append(image)
        if i%16 == 0:
            buffer = np.expand_dims(buffer,0)
            predicted_value = np.argmax(model.predict(buffer))
            cls = labels.iloc[predicted_value]
            print(cls)
            print(cls.iloc[0])
            if predicted_value == 0:
                final_label = "Swiping Left"
                Keyboard.key(Keyboard.VK_MEDIA_NEXT_TRACK)
            elif predicted_value == 1:
                final_label = "Swiping Right"
                Keyboard.key(Keyboard.VK_MEDIA_PREV_TRACK)
            elif predicted_value == 2:
                final_label = "Swiping Down"
                Keyboard.key(Keyboard.VK_VOLUME_DOWN)
            elif predicted_value == 3:
                final_label = "Swiping Up"
                Keyboard.key(Keyboard.VK_VOLUME_UP)
            elif predicted_value == 4:
                final_label = "Pushing Hand Away"
            elif predicted_value == 5:
                final_label = "Pulling Hand In"
            elif predicted_value == 6:
                final_label = "Sliding Two Fingers Left"
            elif predicted_value == 7:
                final_label = "Sliding Two Fingers Right"
            elif predicted_value == 8:
                final_label = "Sliding Two Fingers Down"
                Keyboard.key(Keyboard.VK_VOLUME_DOWN)
            elif predicted_value == 9:
                final_label = "Sliding Two Fingers Up"
                Keyboard.key(Keyboard.VK_VOLUME_UP)
            elif predicted_value == 10:
                final_label = "Pushing Two Fingers Away"
            elif predicted_value == 11:
                final_label = "Pulling Two Fingers In"
            elif predicted_value == 12:
                final_label = "Rolling Hand Forward"
            elif predicted_value == 13:
                final_label = "Rolling Hand Backward"
            elif predicted_value == 14:
                final_label = "Turning Hand Clockwise"
            elif predicted_value == 15:
                final_label = "Turning Hand Counterclockwise"
            elif predicted_value == 16:
                final_label = "Zooming In With Full Hand"
            elif predicted_value == 17:
                final_label = "Zooming Out With Full Hand"
            elif predicted_value == 18:
                final_label = "Zooming In With Two Fingers"
            elif predicted_value == 19:
                final_label = "Zooming Out With Two Fingers"
            elif predicted_value == 20:
                final_label = "Thumb Up"
            elif predicted_value == 21:
                final_label = "Thumb Down"
                Keyboard.key(Keyboard.VK_VOLUME_MUTE)
            elif predicted_value == 22:
                final_label = "Shaking Hand"
            elif predicted_value == 23:
                final_label = "Stop Sign"
                Keyboard.key(Keyboard.VK_MEDIA_PLAY_PAUSE)
            elif predicted_value == 24:
                final_label = "Drumming Fingers"
            elif predicted_value == 25:
                final_label = "No gesture"
            else:
                final_label = "Do Somthing Else"
            cv2.imshow('OUTPUT',frame)
            buffer = []
        i = i+1
        text = "Activity : {}".format(final_label)
        cv2.putText(frame,text,(20,35),cv2.FONT_HERSHEY_SIMPLEX,1.15,(0,255,0),5)
        cv2.imshow('OUTPUT',frame)
    if cv2.waitKey(1) == 27:
        break
vid.release()
cv2.destroyAllWindows()