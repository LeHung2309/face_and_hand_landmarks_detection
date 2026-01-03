import cv2
import time
import threading
from typing import Optional
from config import CameraConfig

class CameraStream:
    """
    Thread-safe camera frame capture.
    """
    def __init__(self, config: CameraConfig):
        self.config = config
        self.capture = cv2.VideoCapture(self.config.device_id)
        if not self.capture.isOpened():
            raise ValueError(f"Error: Could not open video source {self.config.device_id}")
        
        # Set resolution
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.frame: Optional[cv2.typing.MatLike] = None
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        
        # FPS calculation
        self.prev_time = 0.0
        self.current_fps = 0.0

    def start(self):
        """Starts the video capture thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """Continuously grabs frames from the camera."""
        while self.running and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                # If we lose the camera, stop
                self.running = False

    def read(self) -> Optional[cv2.typing.MatLike]:
        """Returns the most recent frame."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def update_fps(self):
        """Updates the FPS counter."""
        current_time = time.time()
        delta = current_time - self.prev_time
        if delta > 0:
            self.current_fps = 1.0 / delta
        self.prev_time = current_time

    def get_fps(self) -> float:
        """Returns the current calculated FPS."""
        return self.current_fps

    def stop(self):
        """Stops the thread and releases resources."""
        self.running = False
        if self.thread:
            self.thread.join()
        self.capture.release()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
