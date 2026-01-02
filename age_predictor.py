import cv2
import threading
import queue
import time
from typing import Optional, Tuple
from config import AgeConfig

class AgePredictor:
    """
    Predicts age using a pre-trained Caffe model in a separate background thread.
    Uses a Last-Result caching strategy.
    """
    def __init__(self, config: AgeConfig):
        self.config = config
        self.net = None
        self.enabled = False
        self.latest_prediction: Optional[str] = None
        
        # Threading components
        self.running = False
        self.thread: Optional[threading.Thread] = None
        # Maxsize=1 ensures we always process the most recent requested frame
        # and drop older requests if the predictor is busy.
        self.input_queue = queue.Queue(maxsize=1)

        if self.config.model_paths.files_exist:
            try:
                self.net = cv2.dnn.readNet(
                    self.config.model_paths.age_model,
                    self.config.model_paths.age_proto
                )
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.enabled = True
                print("Age Prediction Model loaded successfully.")
            except Exception as e:
                print(f"Error loading Age Prediction Model: {e}")
                self.enabled = False
        else:
            print("Warning: Age prediction model files not found. Feature disabled.")

    def start(self):
        """Starts the prediction worker thread."""
        if not self.enabled or self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the prediction worker thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            # clear queue to unblock potential puts
            with self.input_queue.mutex:
                self.input_queue.queue.clear()
            self.thread.join(timeout=1.0)

    def process_frame_async(self, face_image: cv2.typing.MatLike):
        """
        Attempts to submit a frame for processing. 
        Non-blocking: If the worker is busy (queue full), this frame is dropped.
        """
        if not self.enabled or face_image is None or face_image.size == 0:
            return
        
        try:
            # put_nowait raises Full if queue is full
            self.input_queue.put_nowait(face_image)
        except queue.Full:
            # Worker is busy, skip this frame to prevent lag
            pass

    def get_latest_age(self) -> Optional[str]:
        """Returns the most recent age prediction."""
        return self.latest_prediction

    def _worker(self):
        """Background worker loop."""
        while self.running:
            try:
                # Wait briefly for a frame so we can check 'running' flag periodically
                face_image = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            prediction = self._predict_internal(face_image)
            if prediction:
                self.latest_prediction = prediction
            
            self.input_queue.task_done()

    def _predict_internal(self, face_image: cv2.typing.MatLike) -> Optional[str]:
        """Internal synchronous prediction logic."""
        try:
            blob = cv2.dnn.blobFromImage(
                face_image, 
                1.0, 
                (227, 227), 
                self.config.mean_values, 
                swapRB=False
            )
            self.net.setInput(blob)
            preds = self.net.forward()
            i = preds[0].argmax()
            confidence = preds[0][i]
            
            if confidence > self.config.confidence_threshold:
                return self.config.age_buckets[i]
            
        except Exception as e:
            print(f"Age Prediction error: {e}")
            
        return None

    @staticmethod
    def get_face_bbox_normalized(landmarks) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculates normalized bounding box (x, y, w, h) from landmarks with padding.
        Returns values in 0.0-1.0 range.
        """
        if not landmarks:
            return None

        x_min, y_min = 1.0, 1.0
        x_max, y_max = 0.0, 0.0

        for lm in landmarks.landmark:
            if lm.x < x_min: x_min = lm.x
            if lm.x > x_max: x_max = lm.x
            if lm.y < y_min: y_min = lm.y
            if lm.y > y_max: y_max = lm.y

        # Padding (approx 10%)
        padding_x = (x_max - x_min) * 0.1
        padding_y = (y_max - y_min) * 0.1

        x_min = max(0.0, x_min - padding_x)
        y_min = max(0.0, y_min - padding_y)
        x_max = min(1.0, x_max + padding_x)
        y_max = min(1.0, y_max + padding_y)

        return (x_min, y_min, x_max - x_min, y_max - y_min)
