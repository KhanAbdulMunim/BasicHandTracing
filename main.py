import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(0) #capture video object

mpHands = mp.solutions.hands #
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils #drawing on image utilitites

while True:
    success, img = cap.read() #capture video in a frame
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB) #conversion to RGB
    results = hands.process(imgRGB) #process the RGB image to get results

    #print(results.multi_hand_landmarks) #print Data from the results

    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks: #for multiple hand detection
            mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS)

    cv.imshow("Camera", img)
    cv.waitKey(1)
