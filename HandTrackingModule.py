## A library module based on cv2 and mediapipe to detect human hands

## important libraried  included
import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detConfidence = 0.5, trackConfidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detConfidence = detConfidence
        self.trackConfidence = trackConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detConfidence, self.trackConfidence)
        self.mpDrawing = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]

    #detect hands method with drawing option
    def detectHands(self, frame, draw = True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting the frame to gray
        self.results = self.hands.process(frameRGB) #method called from mideapipe to detect hands (needs gray images) 
            
        if self.results.multi_hand_landmarks: 
            for handLMS in self.results.multi_hand_landmarks:
                if draw: # draw hand landmarks
                    self.mpDrawing.draw_landmarks(frame, handLMS, self.mpHands.HAND_CONNECTIONS)
        return frame

    #locating the finger tips of the hand based on the mediapipe documantation
    def findPosition(self, frame, handNum=0, draw=True):
        self.LM_list = [] #landmark list
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNum]

            for id, lm in enumerate(hand.landmark):
                height, width, channel = frame.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                self.LM_list.append([id,cx,cy])
                if draw:
                    cv2.circle(frame, (cx,cy), 5, (255,0,255), cv2.FILLED)
        return self.LM_list

    #method to determine how many fingers are up on a hand
    def fingersUp(self):
        fingers = [] #fingers list (5 items)
        if self.LM_list[self.tipIds[0]][1] < self.LM_list[self.tipIds[0]-1][1]: #the thump is up if its tip is above the middle
                                                                                # of the finger point
            fingers.append(1) # add '1' to the index of the finger if the finger is up
        else:
            fingers.append(0) # add '0' to the index of the finger if the finger is down
 
        for id in range(1, 5): #checking for the other four fingers
            if self.LM_list[self.tipIds[id]][2] < self.LM_list[self.tipIds[id]-2][2]: 
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers



def main():
    video = cv2.VideoCapture(0)
    preTime = 0
    curTime = 0

    detector = handDetector()
    while True:
        ret, frame = video.read()

        frame = detector.detectHands(frame)
        LM_list = detector.findPosition(frame, draw=False) #locating the finger positions

        curTime = time.time() ###
        fps = 1/(curTime-preTime) # Frame per second counting
        preTime = curTime ###
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), font, 3, (0,255,0), 3) #displaying the fps
        cv2.imshow("Live Video", frame) #display the video 

        if cv2.waitKey(1) == ord('q'): 
            break
    video.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()