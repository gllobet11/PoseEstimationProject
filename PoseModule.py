import mediapipe as mp
import cv2
import time
import math

#video: Dip_antic_82.5.mp4
#video: Dip_nou_100.mp4

class POSEDETECTOR():

    def __init__(self,mode=False,complexity=1,smoothland=True,segmentation=False,
                 smoothsegm=True,detectionCon=0.5,trackingCon=0.5):

        self.mode=mode
        self.complexity=complexity
        self.smoothland=smoothland
        self.segmentation=segmentation
        self.smoothsegm=smoothsegm
        self.detectionCon=detectionCon #detection confidence
        self.trackingCon=trackingCon #tracking confidence

        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose= mp.solutions.pose
        self.pose= self.mpPose.Pose(self.mode,self.complexity,self.smoothland,
                               self.segmentation,self.smoothsegm,
                               self.detectionCon,self.trackingCon)

    def findPose(self,img,draw=True):


        imgRGB= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)  # pytype: disable=attribute-error

        if self.results.pose_landmarks:
            # Define DrawingSpec for connections (lines)
            connection_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
                color=(0, 255, 0),
                thickness=2,

            )
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS,
                                           connection_drawing_spec=connection_drawing_spec)

        return img

    def findPosition(self,img,draw=True):

        self.lmList=[]
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape
                #print(id, lm)
                cx,cy=int(lm.x*w),int(lm.y*h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        return self.lmList

    def findAngle(self,img,p1,p2,p3,draw=True):
        #get the landmarks
        x1,y1=self.lmList[p1][1:]
        x2,y2=self.lmList[p2][1:]
        x3,y3=self.lmList[p3][1:] #

        #calculate the angle
        angle=math.degrees(math.atan2(y3-y2,x3-x2)
                           -math.atan2(y1-y2,x1-x2))
        if angle <0:
            angle +=360
        #print(angle)

        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255),3)
            cv2.circle(img,(x1,y1),5,(255,0,0),cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 5, (255, 0, 0), cv2.FILLED)
            cv2.putText(img, str(int(angle)),(x2-20,y2+50),
                            cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
        return angle


def main():
    # Try with a simpler video path first
    cap = cv2.VideoCapture('PoseVideos/Dip_nou_100.mp4')  # or use 0 for webcam

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    pTime = 0
    detector = POSEDETECTOR()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Video ended or failed to read frame.")
            break

        img = detector.findPose(img)
        lmList=detector.findPosition(img)
        # interesting landmarks: hip: 23,24 shoulder: 12,11 elbow: 13,14
        print(lmList[11])
        cv2.circle(img, (lmList[11][1], lmList[11][2]), 10, (0, 0, 255),
                   cv2.FILLED)
        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display FPS
        cv2.putText(img, f"FPS: {int(fps)}", (70, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Pose Detection", img)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()