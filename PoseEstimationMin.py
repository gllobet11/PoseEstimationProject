import mediapipe as mp
import cv2
import time


mpDraw=mp.solutions.drawing_utils
mpPose= mp.solutions.pose
pose= mpPose.Pose(model_complexity=2)
#tracking confidence
#detection confidence

# Define the style for landmarks and connections
landmark_style = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # Green landmarks
connection_style = mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)  # Blue connections

#video: Dip_antic_82.5.mp4
#video: Dip_nou_100.mp4
cap=cv2.VideoCapture('PoseVideos/Dip_nou_100.mp4')
pTime=0

#interesting landmarks: hip: 23,24 shoulder: 12,11 elbow: 13,14

while True:
    success, img = cap.read()
    imgRGB= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS,
                              landmark_drawing_spec=landmark_style,
                              connection_drawing_spec= connection_style)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c=img.shape
            print(id, lm)
            cx,cy=int(lm.x*w),int(lm.y*h)
            cv2.circle(img,(cx,cy),5,(0,255,0),cv2.FILLED)


    #change the framerate
    cTime=time.time()
    fps= 1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img, str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN,3, (255,0,0), 3)
    cv2.imshow("Image", img)

    cv2.waitKey(10)