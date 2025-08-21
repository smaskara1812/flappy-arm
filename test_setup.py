#!/usr/bin/env python3
"""
Test script to verify the setup for Computer Vision Flappy Bird
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    packages = ['cv2', 'mediapipe', 'numpy']
    
    print("Testing package imports...")
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {package}: {e}")
            return False
    return True

def test_camera():
    """Test if camera can be accessed"""
    try:
        import cv2
        print("\nTesting camera access...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not accessible")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from camera")
            return False
        
        print(f"✅ Camera accessible - Frame size: {frame.shape}")
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe pose detection"""
    try:
        import mediapipe as mp
        print("\nTesting MediaPipe pose detection...")
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✅ MediaPipe pose detection initialized successfully")
        pose.close()
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def main():
    print("Computer Vision Flappy Bird - Setup Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test camera
    camera_ok = test_camera()
    
    # Test MediaPipe
    mediapipe_ok = test_mediapipe()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Package Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Camera Access: {'✅ PASS' if camera_ok else '❌ FAIL'}")
    print(f"MediaPipe: {'✅ PASS' if mediapipe_ok else '❌ FAIL'}")
    
    if all([imports_ok, camera_ok, mediapipe_ok]):
        print("\n🎉 All tests passed! You're ready to run the camera module.")
        print("Run: python camera_module.py")
    else:
        print("\n⚠️  Some tests failed. Please check your setup.")
        print("Make sure you have:")
        print("1. Installed all requirements: pip install -r requirements.txt")
        print("2. A working webcam")
        print("3. Proper permissions for camera access")

if __name__ == "__main__":
    main()
