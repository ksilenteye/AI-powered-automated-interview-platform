"""
Real-time Monitoring System using OpenCV and YOLO
Face detection, eye tracking, and phone detection
"""

import cv2
import numpy as np
from ultralytics import YOLO


class MonitoringSystem:
    def __init__(self):
        """Initialize monitoring system with models"""
        try:
            # Load YOLO model for object detection
            self.yolo_model = YOLO('yolov8n.pt')  # Will auto-download if not present
            print("✓ YOLO model loaded successfully")
        except Exception as e:
            print(f" YOLO model loading failed: {e}")
            self.yolo_model = None

        # Load Haar Cascade classifiers for face and eye detection
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            print("✓ OpenCV cascades loaded successfully")
        except Exception as e:
            print(f" Cascade loading failed: {e}")
            self.face_cascade = None
            self.eye_cascade = None

        # COCO dataset class IDs
        self.phone_class_id = 67  # Cell phone
        self.person_class_id = 0   # Person

        # Detection settings
        self.confidence_threshold = 0.5
        self.frame_skip = 3  # Process every 3rd frame for performance
        self.frame_count = 0

    def process_frame(self, frame):
        """Process single video frame for monitoring"""
        if frame is None:
            return None, self._empty_results()

        self.frame_count += 1

        # Initialize results
        results = {
            'face_detected': False,
            'eyes_detected': False,
            'eye_count': 0,
            'phone_detected': False,
            'multiple_people': False,
            'person_count': 0,
            'looking_at_camera': False
        }

        try:
            # Convert frame to correct format
            if isinstance(frame, np.ndarray):
                processed_frame = frame.copy()
            else:
                processed_frame = np.array(frame)

            # Ensure RGB format
            if len(processed_frame.shape) == 2:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
            elif processed_frame.shape[2] == 4:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGBA2BGR)

            # Face and eye detection (lightweight, run every frame)
            if self.face_cascade is not None:
                gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                results = self._detect_face_and_eyes(processed_frame, gray, results)

            # YOLO detection (heavier, run every N frames)
            if self.yolo_model is not None and self.frame_count % self.frame_skip == 0:
                results = self._detect_objects_yolo(processed_frame, results)

            # Draw status overlay
            self._draw_status_overlay(processed_frame, results)

            return processed_frame, results

        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame, self._empty_results()

    def _detect_face_and_eyes(self, frame, gray, results):
        """Detect face and eyes using Haar Cascades"""
        try:
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )

            if len(faces) > 0:
                results['face_detected'] = True

                # Draw rectangles around faces
                for (x, y, w, h) in faces:
                    # Face rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(
                        frame, 'Face', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

                    # Detect eyes within face region
                    if self.eye_cascade is not None:
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = frame[y:y+h, x:x+w]

                        eyes = self.eye_cascade.detectMultiScale(
                            roi_gray,
                            scaleFactor=1.1,
                            minNeighbors=10,
                            minSize=(20, 20)
                        )

                        results['eye_count'] = len(eyes)

                        if len(eyes) >= 2:
                            results['eyes_detected'] = True
                            results['looking_at_camera'] = True

                            # Draw eye rectangles
                            for (ex, ey, ew, eh) in eyes:
                                cv2.rectangle(
                                    roi_color,
                                    (ex, ey), (ex+ew, ey+eh),
                                    (255, 0, 0), 2
                                )
                        elif len(eyes) == 1:
                            # One eye detected - partial detection
                            for (ex, ey, ew, eh) in eyes:
                                cv2.rectangle(
                                    roi_color,
                                    (ex, ey), (ex+ew, ey+eh),
                                    (255, 165, 0), 2
                                )
            else:
                # No face detected - draw warning
                cv2.putText(
                    frame, '⚠ NO FACE DETECTED', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3
                )

        except Exception as e:
            print(f"Face detection error: {e}")

        return results

    def _detect_objects_yolo(self, frame, results):
        """Detect objects using YOLO (phone, multiple people)"""
        try:
            # Run YOLO detection
            yolo_results = self.yolo_model(frame, verbose=False, conf=self.confidence_threshold)

            person_count = 0

            for result in yolo_results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Phone detection
                    if class_id == self.phone_class_id:
                        results['phone_detected'] = True

                        # Draw phone detection box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(
                            frame, f'⚠ PHONE! {confidence:.2f}', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3
                        )

                    # Person counting
                    elif class_id == self.person_class_id:
                        person_count += 1

            results['person_count'] = person_count
            if person_count > 1:
                results['multiple_people'] = True
                cv2.putText(
                    frame, f'⚠ {person_count} PEOPLE DETECTED', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3
                )

        except Exception as e:
            print(f"YOLO detection error: {e}")

        return results

    def _draw_status_overlay(self, frame, results):
        """Draw monitoring status overlay"""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Title
        cv2.putText(
            frame, 'MONITORING STATUS', (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        # Status items
        y_offset = 60
        statuses = [
            ('Face', results['face_detected']),
            ('Eyes', results['eyes_detected']),
            ('Phone', not results['phone_detected']),  # Inverted - no phone is good
        ]

        for name, status in statuses:
            color = (0, 255, 0) if status else (0, 0, 255)
            status_text = 'OK' if status else 'FAIL'

            cv2.putText(
                frame, f'{name}:', (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            cv2.putText(
                frame, status_text, (150, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            y_offset += 30

        # Add frame counter (for debugging)
        cv2.putText(
            frame, f'Frame: {self.frame_count}', (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
        )

    def _empty_results(self):
        """Return empty results structure"""
        return {
            'face_detected': False,
            'eyes_detected': False,
            'eye_count': 0,
            'phone_detected': False,
            'multiple_people': False,
            'person_count': 0,
            'looking_at_camera': False
        }
