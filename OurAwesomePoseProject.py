import time
import cv2
import PoseModule as pm
import pygetwindow as gw

# video=input('What video do you wanna analyze? \n'
#       '1: 50kg Muscle Up side view\n'
#       '2:47.5kg Muscle Up front view\n '
#       '3: 100kg Dip\n')

#video: Dip_Pere_182.5.mp4
#video: MU50.mp4
#video: MU47.5frontal.mp4
#video: Dips_100_corto.mp4
# if video=='1':
#     cap = cv2.VideoCapture('PoseVideos/MU50.mp4')  # or use 0 for webcam
# elif video=='2':
#     cap = cv2.VideoCapture('PoseVideos/MU47.5frontal.mp4')  # or use 0 for webcam
# else :
#     cap = cv2.VideoCapture('PoseVideos/Dips_100_corto.mp4')  # or use 0 for webcam

cap = cv2.VideoCapture('PoseVideos/Dips_100_corto.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")

pTime = 0
detector = pm.POSEDETECTOR()

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Video ended or failed to read frame.")
        break

    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    if len(lmList) !=0:
        print(lmList[13])
        cv2.circle(img, (lmList[13][1], lmList[13][2]), 10, (0, 0, 255),
                   cv2.FILLED)
    # interesting landmarks: hip: 23,24 shoulder: 12,11 elbow: 13,14


    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS
    cv2.putText(img, f"FPS: {int(fps)}", (70, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Pose Detection", img)


    # Exit on 'q' key
    print('press q to exit')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

