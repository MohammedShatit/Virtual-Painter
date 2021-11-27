##Virtual Painter application made for learning goals
# use Two fingers to select color and one to draw 


## including important libraries
import cv2
import mediapipe as mp
import HandTrackingModule as ht # hand tracking module that includes some methods to detect fingers tips and how many are up
import numpy as np
import os
import time

colorList = os.listdir("Header") #including the selection image
overlayList = []

for img_path in colorList: #adding 8 variations of the header based on the color/item selected 
    image = cv2.imread(f'Header/{img_path}')
    overlayList.append(image)

overlay = overlayList[5] #first color in the list as the default
color = (0,0,255)
brushThickness = 15 #drawing brush size
xprev, yprev = 0, 0 # the x and y of the previous position of the index finger (used for drawing technique)
drawing_frame = np.zeros((720,1280,3), np.uint8) #the image where we draw

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = ht.handDetector(detConfidence=0.85) #initializing the detector object

while True:
    ret, frame = cap.read() #reading from the recording device
    frame = cv2.flip(frame, 1)

    frame = detector.detectHands(frame, draw=False) #detecting hands
    lmList = detector.findPosition(frame, draw=False) #finding position of the fingers

    if len(lmList) != 0:

        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

        fingers = detector.fingersUp() ##detect which fingers are up
        #print(fingers)

        if fingers[1] and fingers[2]: #detect if the index and the middle fingers are both up
            xprev, yprev = 0, 0
            #print("Selection mode")
            if y1 < 125: # detect if we are in the header range
                if 20<x1<150: #check the range of color we are in from the list and select it
                    overlay = overlayList[5]
                    color = (0,0,255)
                    brushThickness = 15
                elif 160<x1<290:
                    overlay = overlayList[4]
                    color = (0,119,255)
                    brushThickness = 15
                elif 300<x1<430:
                    overlay = overlayList[7]
                    color = (0,205,255)
                    brushThickness = 15
                elif 440<x1<570:
                    overlay = overlayList[2]
                    color = (12,111,35)
                    brushThickness = 15
                elif 580<x1<710:
                    overlay = overlayList[0]
                    color = (255,0,0)
                    brushThickness = 15
                elif 720<x1<850:
                    overlay = overlayList[3]
                    color = (111,12,68)
                    brushThickness = 15
                elif 860<x1<990:
                    overlay = overlayList[6]
                    color = (209,55,158)
                    brushThickness = 15
                elif 1020<x1<1220:
                    overlay = overlayList[1]
                    color = (0,0,0)
                    brushThickness = 60
            cv2.rectangle(frame, (x1,y1-20),(x2,y2+20),color,cv2.FILLED) #draw a rectangle of the selected color in between the 
                                                                         #index and middle finger

        if fingers[1] and fingers[2] == False: # detect if the index is up and the middle finger is down
            cv2.circle(frame, (x1,y1), 15, color, cv2.FILLED) # draw a circle on the index tip to indicate where we are drawing
            if xprev == 0 and yprev == 0: #if we are just starting to draw the deafult position to draw from is the
                                          #corner of the frame (we don't want that) 
                xprev, yprev = x1, y1 #select thje previous position instead

            # the draing technique is  to draw a line from the previous position to the current position
            cv2.line(frame, (xprev,yprev), (x1,y1), color, brushThickness)
            cv2.line(drawing_frame, (xprev,yprev), (x1,y1), color, brushThickness)


            xprev, yprev = x1, y1 # setting the current position of the fingers as the previous position
    #draing on the inversed frame in Gray and reversing back
    frameGray = cv2.cvtColor(drawing_frame, cv2.COLOR_BGR2GRAY)
    _, frame_inversed = cv2.threshold(frameGray, 50, 255, cv2.THRESH_BINARY_INV)
    frame_inversed = cv2.cvtColor(frame_inversed, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, frame_inversed)
    frame = cv2.bitwise_or(frame, drawing_frame)

    frame[0:125, 0:1280] = overlay #displaying the header
    cv2.imshow("Live Video", frame)
    if cv2.waitKey(1) == ord('q'): #use 'q' to end the application 
        break

cap.release()
cv2.destroyAllWindows()
