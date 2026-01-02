import cv2
import mediapipe as mp
import threading
import queue
import time
from typing import NamedTuple, Optional
from config import MediaPipeConfig

class HolisticDetector:
    """
    Async Wrapper for MediaPipe Holistic solution.
    Runs the heavy inference in a background thread.
    """
    def __init__(self, config: MediaPipeConfig):
        self.config = config
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        
        self.face_connection_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 255), thickness=1, circle_radius=1
        )

        # Threading
        self.input_queue = queue.Queue(maxsize=1)
        self.latest_results: Optional[NamedTuple] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock() # For safe access to latest_results

    def start(self):
        """Starts the holistic inference thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the inference thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            with self.input_queue.mutex:
                self.input_queue.queue.clear()
            self.thread.join(timeout=1.0)

    def process_async(self, frame: cv2.typing.MatLike):
        """
        Submits a frame for holistic processing.
        Non-blocking: drops frame if worker is busy.
        """
        if frame is None:
            return
        
        try:
            self.input_queue.put_nowait(frame)
        except queue.Full:
            pass

    def get_latest_results(self) -> Optional[NamedTuple]:
        """Returns the most recent inference results."""
        with self.lock:
            return self.latest_results

    def _worker(self):
        """Background worker loop. Initializes MP here for thread safety."""
        # Initialize MediaPipe Holistic *inside* the thread
        holistic = self.mp_holistic.Holistic(
            static_image_mode=self.config.static_image_mode,
            model_complexity=self.config.model_complexity,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            refine_face_landmarks=self.config.refine_face_landmarks
        )

        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Process
            # MediaPipe expects RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True

            # Update results
            with self.lock:
                self.latest_results = results
            
            self.input_queue.task_done()

        holistic.close()

    def draw_landmarks(self, frame: cv2.typing.MatLike, results: NamedTuple) -> None:
        """
        Draws landmarks on the frame using the provided results.
        Runs on Main Thread.
        """
        if results is None:
            return

        # Face (Optimized)
        if self.config.enable_face and results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                None,
                self.face_connection_spec
            )

        # Pose
        if self.config.enable_pose and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Hands
        if self.config.enable_hands:
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
