import cv2
import numpy as np
from camera_module import PoseAndHandDetector

class CameraComponent:
    def __init__(self, camera_width, camera_height):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.detector = PoseAndHandDetector()
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
    def get_frame(self):
        """Get a frame from the camera"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)  # Mirror effect
    
    def process_frame(self, frame):
        """Process frame for pose and hand detection"""
        pose_results, hand_results = self.detector.detect_pose_and_hands(frame)
        gestures = self.detector.analyze_gestures(pose_results, hand_results)
        return pose_results, hand_results, gestures
    
    def render(self, canvas, offset_x, offset_y):
        """Render the camera view on the provided canvas"""
        # Get and process camera frame
        frame = self.get_frame()
        if frame is None:
            return
        
        pose_results, hand_results, gestures = self.process_frame(frame)
        
        # Draw detection overlays on the frame
        self.detector.draw_pose_landmarks(frame, pose_results)
        self.detector.draw_hand_landmarks(frame, hand_results)
        self.detector.draw_gesture_info(frame, gestures)
        
        # Resize frame to fit camera area
        resized_frame = cv2.resize(frame, (self.camera_width, self.camera_height))
        
        # Place the camera frame in the designated area
        canvas[offset_y:offset_y + self.camera_height, 
               offset_x:offset_x + self.camera_width] = resized_frame
        
        # Draw border around camera view
        cv2.rectangle(canvas, (offset_x - 2, offset_y - 2), 
                     (offset_x + self.camera_width + 2, offset_y + self.camera_height + 2), 
                     (255, 255, 255), 2)
    
    def get_flap_count(self):
        """Get current flap count from detector"""
        flap_counters = self.detector.get_flap_counters()
        return flap_counters['both_flaps']
    
    def reset_counters(self):
        """Reset flap counters"""
        self.detector.reset_flap_counters()
    
    def cleanup(self):
        """Clean up camera resources"""
        self.cap.release()
        self.detector.pose.close()
        self.detector.hands.close()
