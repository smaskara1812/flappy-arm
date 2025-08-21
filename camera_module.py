import cv2
import mediapipe as mp
import numpy as np
import time

class PoseAndHandDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Initialize hand detection
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2  # Detect both hands
        )
        
        # Custom pose connections without wrist connections
        self.custom_pose_connections = [
            # Face connections
            (0, 1), (1, 2), (2, 3), (3, 7),  # Nose to left eye
            (0, 4), (4, 5), (5, 6), (6, 8),  # Nose to right eye
            (9, 10),  # Mouth
            
            # Upper body connections (excluding wrists)
            (11, 12),  # Shoulders
            (11, 13), (13, 15),  # Left arm (without wrist)
            (12, 14), (14, 16),  # Right arm (without wrist)
            
            # Lower body connections
            (11, 23), (12, 24),  # Shoulders to hips
            (23, 24),  # Hips
            (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
            (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
        ]
        
        # Landmark indices for key body parts
        self.pose_landmarks = {
            'nose': 0,
            'left_shoulder': 12,
            'right_shoulder': 11,
            'left_elbow': 14,
            'right_elbow': 13,
            # 'left_wrist': 16,
            # 'right_wrist': 15,
            'left_hip': 24,
            'right_hip': 23
        }
        
        # Hand landmark names for reference
        self.hand_landmarks = {
            'wrist': 0,
            'thumb_tip': 4,
            'index_tip': 8,
            'middle_tip': 12,
            'ring_tip': 16,
            'pinky_tip': 20
        }
        
        # Gesture recognition settings
        self.gesture_history = {
            'left_arm': [],
            'right_arm': [],
            'left_hand': [],
            'right_hand': []
        }
        self.history_length = 8   # Reduced for faster response
        self.movement_threshold = 0.06  # More sensitive to small movements
        self.flap_threshold = 0.08  # Lower threshold for vigorous flapping
        
        # Gesture states
        self.current_gestures = {
            'both_flap': False,
            'hands_up': False,
            'hands_down': False
        }
        
        # Flap counting
        self.flap_counters = {
            'both_flaps': 0,
            'total_flaps': 0
        }
        
        # Flap detection state
        self.flap_states = {
            'both_flap_active': False,
            'both_flap_cooldown': 0,
            'last_flap_direction': 0  # 1 for up, -1 for down, 0 for neutral
        }
        
        # Cooldown settings (frames) - Reduced for vigorous flapping
        self.flap_cooldown_frames = 8  # Shorter cooldown for faster flapping
        
    def detect_pose_and_hands(self, image):
        """Detect both pose and hand landmarks"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process pose
        pose_results = self.pose.process(image_rgb)
        
        # Process hands
        hand_results = self.hands.process(image_rgb)
        
        return pose_results, hand_results
    
    def analyze_gestures(self, pose_results, hand_results):
        """Analyze gestures from pose and hand data"""
        if pose_results.pose_landmarks:
            self._update_arm_history(pose_results.pose_landmarks)
            self._detect_arm_gestures()
        
        if hand_results.multi_hand_landmarks:
            self._update_hand_history(hand_results.multi_hand_landmarks)
            self._detect_hand_gestures()
        
        return self.current_gestures
    
    def _update_arm_history(self, landmarks):
        """Update arm movement history"""
        # Get arm positions (shoulder to elbow)
        left_arm_pos = self._get_arm_position(landmarks, 'left')
        right_arm_pos = self._get_arm_position(landmarks, 'right')
        
        # Update history
        self.gesture_history['left_arm'].append(left_arm_pos)
        self.gesture_history['right_arm'].append(right_arm_pos)
        
        # Keep only recent history
        if len(self.gesture_history['left_arm']) > self.history_length:
            self.gesture_history['left_arm'].pop(0)
        if len(self.gesture_history['right_arm']) > self.history_length:
            self.gesture_history['right_arm'].pop(0)
    
    def _get_arm_position(self, landmarks, side):
        """Get arm position (shoulder to elbow)"""
        if side == 'left':
            shoulder = landmarks.landmark[12]  # left_shoulder
            elbow = landmarks.landmark[14]     # left_elbow
        else:
            shoulder = landmarks.landmark[11]  # right_shoulder
            elbow = landmarks.landmark[13]     # right_elbow
        
        # Return average position of arm
        return {
            'x': (shoulder.x + elbow.x) / 2,
            'y': (shoulder.y + elbow.y) / 2,
            'z': (shoulder.z + elbow.z) / 2,
            'visibility': min(shoulder.visibility, elbow.visibility)
        }
    
    def _update_hand_history(self, multi_hand_landmarks):
        """Update hand movement history"""
        # Reset hand history
        self.gesture_history['left_hand'] = []
        self.gesture_history['right_hand'] = []
        
        for hand_landmarks in multi_hand_landmarks:
            # Determine if left or right hand based on position
            wrist = hand_landmarks.landmark[0]
            hand_pos = {'x': wrist.x, 'y': wrist.y, 'z': wrist.z, 'visibility': wrist.visibility}
            
            # Simple left/right detection based on x position
            if wrist.x < 0.5:  # Left side of frame
                self.gesture_history['left_hand'].append(hand_pos)
            else:  # Right side of frame
                self.gesture_history['right_hand'].append(hand_pos)
    
    def _detect_arm_gestures(self):
        """Detect both arms flapping gesture (count a complete up-down or down-up as one flap)"""
        # Reset gesture states
        self.current_gestures['both_flap'] = False

        # Update cooldown
        if self.flap_states['both_flap_cooldown'] > 0:
            self.flap_states['both_flap_cooldown'] -= 1

        # Check if both arms are flapping
        left_flap_detected = False
        right_flap_detected = False
        left_direction = 0
        right_direction = 0

        # Check left arm flapping
        if len(self.gesture_history['left_arm']) >= 4:
            left_movement = self._calculate_movement(self.gesture_history['left_arm'])
            left_direction = self._detect_flap_direction(self.gesture_history['left_arm'])
            left_velocity = self._calculate_velocity(self.gesture_history['left_arm'])

            # More sensitive detection for vigorous flapping
            if ((left_movement > self.flap_threshold and left_direction != 0) or
                (left_velocity > 0.05 and left_direction != 0)):  # High velocity detection
                left_flap_detected = True

        # Check right arm flapping
        if len(self.gesture_history['right_arm']) >= 4:
            right_movement = self._calculate_movement(self.gesture_history['right_arm'])
            right_direction = self._detect_flap_direction(self.gesture_history['right_arm'])
            right_velocity = self._calculate_velocity(self.gesture_history['right_arm'])

            # More sensitive detection for vigorous flapping
            if ((right_movement > self.flap_threshold and right_direction != 0) or
                (right_velocity > 0.05 and right_direction != 0)):  # High velocity detection
                right_flap_detected = True

        # Detect both arms flapping (direction-based, count only on downward direction change)
        if left_flap_detected and right_flap_detected:
            self.current_gestures['both_flap'] = True

            # Only count a flap if direction changed to downward (-1)
            # Use the direction of the arms (if both are same direction and not neutral)
            if left_direction == right_direction and left_direction != 0:
                last_dir = self.flap_states.get('last_flap_direction', 0)
                # Only increment if direction is -1 (down) and different from last direction, and cooldown is satisfied
                if (left_direction == -1 and left_direction != last_dir and self.flap_states['both_flap_cooldown'] == 0):
                    self.flap_counters['both_flaps'] += 1
                    self.flap_counters['total_flaps'] += 1
                    self.flap_states['both_flap_cooldown'] = self.flap_cooldown_frames
                # Update last direction
                self.flap_states['last_flap_direction'] = left_direction
            # If not both in same direction, don't update last_flap_direction

            self.flap_states['both_flap_active'] = True
        else:
            self.flap_states['both_flap_active'] = False
    
    def _detect_hand_gestures(self):
        """Detect hand-based gestures"""
        # Check hands up/down based on hand positions
        left_hands = self.gesture_history['left_hand']
        right_hands = self.gesture_history['right_hand']
        
        # Calculate average hand positions
        avg_left_y = np.mean([h['y'] for h in left_hands]) if left_hands else 0.5
        avg_right_y = np.mean([h['y'] for h in right_hands]) if right_hands else 0.5
        
        # Determine hands up/down (lower y = higher position)
        hands_up_threshold = 0.4
        hands_down_threshold = 0.6
        
        if avg_left_y < hands_up_threshold and avg_right_y < hands_up_threshold:
            self.current_gestures['hands_up'] = True
            self.current_gestures['hands_down'] = False
        elif avg_left_y > hands_down_threshold and avg_right_y > hands_down_threshold:
            self.current_gestures['hands_up'] = False
            self.current_gestures['hands_down'] = True
        else:
            self.current_gestures['hands_up'] = False
            self.current_gestures['hands_down'] = False
    
    def _calculate_movement(self, history):
        """Calculate movement magnitude from history"""
        if len(history) < 2:
            return 0
        
        # Calculate total movement over recent frames
        total_movement = 0
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]
            
            # Calculate Euclidean distance
            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            movement = np.sqrt(dx*dx + dy*dy)
            total_movement += movement
        
        return total_movement
    
    def _calculate_velocity(self, history):
        """Calculate movement velocity (speed) for vigorous flapping detection"""
        if len(history) < 3:
            return 0
        
        # Calculate velocity over last 3 frames
        recent = history[-3:]
        total_distance = 0
        
        for i in range(1, len(recent)):
            prev = recent[i-1]
            curr = recent[i]
            
            # Calculate distance between consecutive frames
            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            distance = np.sqrt(dx*dx + dy*dy)
            total_distance += distance
        
        # Velocity = total distance / number of frame intervals
        velocity = total_distance / (len(recent) - 1)
        return velocity
    
    def _detect_flap_direction(self, history):
        """Detect the direction of flap movement (up/down)"""
        if len(history) < 2:
            return 0
        
        # Get recent positions (reduced from 3 to 2 for faster response)
        recent = history[-2:]
        
        # Calculate vertical movement trend
        y_positions = [pos['y'] for pos in recent]
        
        # Check if movement is significant (reduced threshold for vigorous flapping)
        y_range = max(y_positions) - min(y_positions)
        if y_range < 0.01:  # Reduced threshold for more sensitivity
            return 0
        
        # Determine direction based on trend (for 2 frames)
        if y_positions[0] > y_positions[1]:
            return 1  # Moving up (decreasing y)
        elif y_positions[0] < y_positions[1]:
            return -1  # Moving down (increasing y)
        else:
            return 0  # No clear direction
    
    def get_flap_counters(self):
        """Get current flap counters"""
        return self.flap_counters.copy()
    
    def reset_flap_counters(self):
        """Reset all flap counters"""
        self.flap_counters = {
            'both_flaps': 0,
            'total_flaps': 0
        }
    
    def draw_pose_landmarks(self, image, results):
        """Draw pose landmarks on the image"""
        if results.pose_landmarks:
            # Draw custom pose connections (no wrists)
            self._draw_custom_pose_connections(image, results.pose_landmarks)
            
            # Draw key pose landmarks with custom colors
            self._draw_key_pose_landmarks(image, results.pose_landmarks)
    
    def _draw_custom_pose_connections(self, image, landmarks):
        """Draw custom pose connections without wrist landmarks"""
        h, w, _ = image.shape
        
        # Draw only the connections we want (excluding wrists)
        for connection in self.custom_pose_connections:
            start_idx, end_idx = connection
            
            # Skip if either landmark has low visibility
            if (landmarks.landmark[start_idx].visibility < 0.5 or 
                landmarks.landmark[end_idx].visibility < 0.5):
                continue
            
            # Get landmark positions
            start_x = int(landmarks.landmark[start_idx].x * w)
            start_y = int(landmarks.landmark[start_idx].y * h)
            end_x = int(landmarks.landmark[end_idx].x * w)
            end_y = int(landmarks.landmark[end_idx].y * h)
            
            # Draw connection line
            cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
    
    def draw_hand_landmarks(self, image, results):
        """Draw hand landmarks on the image"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw all hand landmarks
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
                )
                
                # Draw key hand landmarks with custom colors
                self._draw_key_hand_landmarks(image, hand_landmarks)
    
    def draw_gesture_info(self, image, gestures):
        """Draw gesture information on the image"""
        h, w, _ = image.shape
        
        # Get flap counters
        flap_counters = self.get_flap_counters()
        
        # Gesture status colors
        active_color = (0, 255, 0)    # Green for active
        inactive_color = (128, 128, 128)  # Gray for inactive
        
        # Draw gesture status
        y_offset = 150
        for gesture_name, is_active in gestures.items():
            color = active_color if is_active else inactive_color
            status = "ACTIVE" if is_active else "inactive"
            
            cv2.putText(image, f"{gesture_name.replace('_', ' ').title()}: {status}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        # Draw flap counters
        y_offset += 20
        cv2.putText(image, "--- FLAP COUNTERS ---", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        y_offset += 30
        
        cv2.putText(image, f"Both Arms Flaps: {flap_counters['both_flaps']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        y_offset += 25
        
        cv2.putText(image, f"Total Flaps: {flap_counters['total_flaps']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw movement indicators
        if gestures['both_flap']:
            cv2.putText(image, "BOTH ARMS FLAPPING!", (w//2 - 120, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        
        # Draw instructions
        cv2.putText(image, "Flap both arms to control the bird!", (10, h - 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "Move both arms up/down together", (10, h - 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "Press 'r' to reset counters", (10, h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "Press 'q' to quit", (10, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _draw_key_pose_landmarks(self, image, landmarks):
        """Draw key pose landmarks with custom colors and labels"""
        h, w, _ = image.shape
        
        # Colors for different body parts
        colors = {
            'face': (0, 255, 0),      # Green for face
            'shoulders': (255, 0, 0),  # Blue for shoulders
            'elbows': (0, 0, 255),     # Red for elbows
            'hips': (255, 0, 255)      # Magenta for hips
        }
        
        # Draw key pose landmarks
        for name, idx in self.pose_landmarks.items():
            if landmarks.landmark[idx].visibility > 0.5:
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                
                # Determine color based on landmark type
                if 'shoulder' in name:
                    color = colors['shoulders']
                elif 'elbow' in name:
                    color = colors['elbows']
                elif 'hip' in name:
                    color = colors['hips']
                else:
                    color = colors['face']
                
                # Draw circle and label
                cv2.circle(image, (x, y), 8, color, -1)
                cv2.putText(image, name.replace('_', ' ').title(), 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 2)
    
    def _draw_key_hand_landmarks(self, image, landmarks):
        """Draw key hand landmarks with custom colors"""
        h, w, _ = image.shape
        
        # Colors for hand landmarks
        hand_colors = {
            'wrist': (255, 255, 255),      # White for wrist
            'thumb_tip': (255, 0, 0),      # Blue for thumb
            'index_tip': (0, 255, 0),      # Green for index
            'middle_tip': (0, 0, 255),     # Red for middle
            'ring_tip': (255, 255, 0),     # Cyan for ring
            'pinky_tip': (255, 0, 255)     # Magenta for pinky
        }
        
        # Draw key hand landmarks
        for name, idx in self.hand_landmarks.items():
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            
            color = hand_colors[name]
            
            # Draw circle and label
            cv2.circle(image, (x, y), 6, color, -1)
            cv2.putText(image, name.replace('_', ' ').title(), 
                       (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, color, 1)
    
    def get_pose_positions(self, results):
        """Get positions of key pose landmarks"""
        positions = {}
        
        if results.pose_landmarks:
            for name, idx in self.pose_landmarks.items():
                landmark = results.pose_landmarks.landmark[idx]
                if landmark.visibility > 0.5:
                    positions[name] = (landmark.x, landmark.y, landmark.z)
        
        return positions
    
    def get_hand_positions(self, results):
        """Get positions of hand landmarks"""
        hand_positions = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = {}
                for name, idx in self.hand_landmarks.items():
                    landmark = hand_landmarks.landmark[idx]
                    hand_data[name] = (landmark.x, landmark.y, landmark.z)
                hand_positions.append(hand_data)
        
        return hand_positions

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Increased FPS for better tracking
    
    # Initialize detector
    detector = PoseAndHandDetector()
    
    # Performance tracking
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0.0  # Initialize fps variable
    
    print("Computer Vision Flappy Bird - Pose & Hand Detection")
    print("Press 'q' to quit")
    print("Detecting: Face, Shoulders, Elbows, Hips + Hand Details")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect pose and hands
        pose_results, hand_results = detector.detect_pose_and_hands(frame)
        
        # Analyze gestures
        gestures = detector.analyze_gestures(pose_results, hand_results)
        
        # Draw pose landmarks
        detector.draw_pose_landmarks(frame, pose_results)
        
        # Draw hand landmarks
        detector.draw_hand_landmarks(frame, hand_results)
        
        # Draw gesture information
        detector.draw_gesture_info(frame, gestures)
        
        # Get landmark positions for future gesture recognition
        pose_positions = detector.get_pose_positions(pose_results)
        hand_positions = detector.get_hand_positions(hand_results)
        
        # Calculate and display FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        # Display FPS and status
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display detected landmarks count
        pose_count = len(pose_positions)
        hand_count = len(hand_positions)
        cv2.putText(frame, f"Pose Landmarks: {pose_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Hands Detected: {hand_count}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Computer Vision Flappy Bird - Pose & Hand Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset_flap_counters()
            print("Flap counters reset!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.pose.close()
    detector.hands.close()

if __name__ == "__main__":
    main()