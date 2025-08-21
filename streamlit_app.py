import streamlit as st
import cv2
import numpy as np
import time
from camera_module import PoseAndHandDetector
from game_component import GameComponent

def main():
    st.title("🎮 Computer Vision Flappy Bird")
    st.write("Flap your arms to control the bird!")
    
    # Initialize components
    if 'detector' not in st.session_state:
        st.session_state.detector = PoseAndHandDetector()
    if 'game' not in st.session_state:
        st.session_state.game = GameComponent(800, 600)
    
    # Camera input
    camera_input = st.camera_input("Take a picture to start")
    
    if camera_input is not None:
        # Convert to OpenCV format
        bytes_data = camera_input.getvalue()
        cv2_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Process image
        pose_results, hand_results = st.session_state.detector.detect_pose_and_hands(cv2_image)
        gestures = st.session_state.detector.analyze_gestures(pose_results, hand_results)
        
        # Get flap count
        flap_counters = st.session_state.detector.get_flap_counters()
        current_flaps = flap_counters['both_flaps']
        
        # Update game
        st.session_state.game.update_physics(current_flaps)
        
        # Create game frame
        game_frame = np.zeros((600, 800, 3), dtype=np.uint8)
        st.session_state.game.render(game_frame, 0, 0)
        
        # Display game
        st.image(game_frame, channels="BGR", caption="Flappy Bird Game")
        
        # Display controls
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score", int(st.session_state.game.score))
        with col2:
            st.metric("Flaps Detected", current_flaps)
        with col3:
            if st.button("Reset Game"):
                st.session_state.game.reset()
                st.session_state.detector.reset_flap_counters()
                st.rerun()

if __name__ == "__main__":
    main()
