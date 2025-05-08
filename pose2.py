import mediapipe as mp
import cv2
import time


class POSEDETECTOR():
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.mode = mode
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)

        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    self.mpPose.POSE_CONNECTIONS
                )
        return img
   # def getPosition(self,img,draw=True):




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