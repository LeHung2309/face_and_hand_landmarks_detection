import cv2
import time
import mediapipe as mp
from typing import Optional, Tuple, NamedTuple

class HolisticDetector:
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.holistic = self.mp_holistic.Holistic(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.face_landmark_spec = self.mp_drawing.DrawingSpec(
            color=(255, 0, 255), thickness=1, circle_radius=1
        )
        self.face_connection_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 255), thickness=1, circle_radius=1
        )

    def process_frame(self, frame: cv2.typing.MatLike) -> NamedTuple:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        
        return results

    def draw_landmarks(self, frame: cv2.typing.MatLike, results: NamedTuple) -> None:
        if results is None:
            return

        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.face_connection_spec
            )

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )

        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )

    def close(self):
        self.holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def main():
    WIDTH, HEIGHT = 800, 600
    CAP_DEVICE = 0

    capture = cv2.VideoCapture(CAP_DEVICE)
    if not capture.isOpened():
        print(f"Error: Could not open video source {CAP_DEVICE}")
        return

    previous_time = 0
    current_time = 0

    with HolisticDetector() as detector:
        print("Holistic Detector started. Press 'q' to exit.")
        
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.resize(frame, (WIDTH, HEIGHT))

            results = detector.process_frame(frame)

            detector.draw_landmarks(frame, results)

            current_time = time.time()
            fps = 1 / (current_time - previous_time) if (current_time - previous_time) > 0 else 0
            previous_time = current_time

            cv2.putText(frame, f"{int(fps)} FPS", (10, 70), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Facial and Hand Landmarks", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()