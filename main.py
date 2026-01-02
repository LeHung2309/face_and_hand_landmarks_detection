import cv2
import time
from config import AppConfig
from camera_utils import CameraStream
from detectors import HolisticDetector
from age_predictor import AgePredictor

def main():
    # Optimization flags
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    # Initialize Configuration
    config = AppConfig()

    # Initialize Components
    try:
        camera = CameraStream(config.camera)
        detector = HolisticDetector(config.mediapipe)
        age_predictor = AgePredictor(config.age)
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    # Start Background Threads
    print("Starting AI threads...")
    detector.start()
    age_predictor.start()
    camera.start()
    
    print(f"{config.window_name} started. Press 'q' to exit.")
    
    # Warmup
    time.sleep(1.0)

    try:
        while True:
            # 1. Capture Frame (High Res)
            frame_display = camera.read()
            if frame_display is None:
                if not camera.running:
                    break
                time.sleep(0.001)
                continue

            h, w, _ = frame_display.shape

            # 2. Prepare Frame for AI (Downscale)
            # We use a smaller copy for the Holistic model to speed up inference
            scale = config.processing_width / w
            proc_w = config.processing_width
            proc_h = int(h * scale)
            
            frame_process = cv2.resize(frame_display, (proc_w, proc_h))

            # 3. Submit to Holistic Detector (Async - Fire & Forget)
            # Use .copy() to ensure thread safety as frame_process might be modified or reused
            detector.process_async(frame_process.copy())

            # 4. Retrieve Latest Available Results (Non-blocking)
            results = detector.get_latest_results()
            current_age = age_predictor.get_latest_age()

            # 5. Handle Age Prediction
            # If we have a face detection, we can try to update the age
            if results and results.face_landmarks:
                # Calculate bbox from landmarks
                # We use the normalized coordinates, so they work on the current High Res frame
                bbox_norm = age_predictor.get_face_bbox_normalized(results.face_landmarks)
                
                if bbox_norm:
                    nx, ny, nw, nh = bbox_norm
                    
                    # Convert to pixel coordinates of display frame
                    x = int(nx * w)
                    y = int(ny * h)
                    bw = int(nw * w)
                    bh = int(nh * h)
                    
                    # Ensure within bounds
                    x = max(0, x)
                    y = max(0, y)
                    bw = min(w - x, bw)
                    bh = min(h - y, bh)

                    if bw > 0 and bh > 0:
                        # Extract face crop from HIGH RES frame for best age accuracy
                        face_img = frame_display[y:y+bh, x:x+bw].copy()
                        # Submit to Age Predictor (Async - Fire & Forget)
                        age_predictor.process_frame_async(face_img)

            # 6. Draw Results on Display Frame
            # We draw whatever results we have (even if slightly old)
            detector.draw_landmarks(frame_display, results)
            
            # Draw Age Text
            if current_age and results and results.face_landmarks:
                 bbox_norm = age_predictor.get_face_bbox_normalized(results.face_landmarks)
                 if bbox_norm:
                    nx, ny, _, _ = bbox_norm
                    x = int(nx * w)
                    y = int(ny * h)
                    text_pos_y = max(30, y - 10)
                    cv2.putText(frame_display, f"Age: {current_age}", (x, text_pos_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2, cv2.LINE_AA)

            # 7. FPS Counter
            camera.update_fps()
            fps = camera.get_fps()
            cv2.putText(frame_display, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 8. Display
            cv2.imshow(config.window_name, frame_display)

            # 9. Exit Control
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Cleanup
        print("Stopping threads...")
        camera.stop()
        detector.stop()
        age_predictor.stop()
        cv2.destroyAllWindows()
        print("Application closed.")

if __name__ == "__main__":
    main()
