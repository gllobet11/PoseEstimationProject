import cv2
import numpy as np
import time

from numpy.ma.core import angle

import PoseModule as pm



#video: Dip_Pere_182.5.mp4
cap= cv2.VideoCapture('PoseVideos/Dip_Pere_182.5.mp4')

detector=pm.POSEDETECTOR()
count=0
dir=0
pTime=0
while True:
    success, img = cap.read()
    #img = cv2.resize(img, (480, 848))
    #img = cv2.imread('AI trainer/test.jpg')
    img=detector.findPose(img,False)
    lmList=detector.findPosition(img,False)
    print(lmList)
    if len(lmList) != 0:
        #Left arm
        angle=detector.findAngle(img,15,13,11)#wrist elbow shoulder
        #Right arm
        #angle=detector.findAngle(img, 16, 14, 12)  # wrist elbow shoulder
        per=np.interp(angle,(90,160),(0,100))
        print(angle,per)

        #check for the dip rep
        if per == 100:
            if dir==0:
                count+=0.5
                dir=1
        if per ==0:
            if dir ==1:
                count+=0.5
                dir=0
        print (count)

        bar = np.interp(angle, (90, 160), (720, 360))  # más ángulo = barra más alta
        cv2.rectangle(img, (50, int(bar)), (100, 720), (0, 255, 0), cv2.FILLED)

        cv2.putText(img,str(int(count)),(45,670),cv2.FONT_HERSHEY_PLAIN,5,
        (255,0,0),5)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)

    # interesting landmarks: hip: 23,24 shoulder: 12,11 elbow: 13,14
    cv2.imshow('Image',img)
    cv2.waitKey(1)


