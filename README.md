# Computer Vision Flappy Bird

A computer vision-based Flappy Bird game that uses gesture recognition to control the ball's bouncing motion.

## Features

- Real-time camera feed with pose and hand detection
- Gesture recognition for "flapping" motion using both arms
- Computer vision-based game control
- Face, shoulder, elbow, and hand tracking
- Two game versions: Simple overlay and Modular split-screen

## Setup Instructions

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the basic camera module to test detection:
```bash
python camera_module.py
```

3. Run the simple game version (overlay on camera):
```bash
python flappy_ball_game.py
```

4. Run the modular game version (clean split-screen):
```bash
python main_game.py
```

## Controls

- **Flap both arms** to make the ball bounce up
- Press 'q' to quit the application
- Press 'r' to reset the game

## Project Structure

### Core Files
- `camera_module.py` - Main camera detection and gesture recognition
- `flappy_ball_game.py` - Simple game version (overlay on camera)
- `main_game.py` - Modular game version (split-screen layout)

### Modular Components
- `game_component.py` - Game physics and rendering
- `camera_component.py` - Camera feed and detection display

### Utility Files
- `test_setup.py` - Environment testing script
- `requirements.txt` - Python dependencies

## Development Phases

### Phase 1: Basic Camera Module ✅
- Camera feed with pose detection
- Landmark visualization
- Basic gesture recognition setup

### Phase 2: Gesture Recognition ✅
- Flapping motion detection
- Both arms gesture recognition
- Flap counting system

### Phase 3: Game Integration ✅
- Ball physics with gravity and bouncing
- Real-time control integration
- Score tracking

### Phase 4: UI/UX Enhancement ✅
- Split-screen layout
- Modular component architecture
- Clean game interface

## Technical Details

- **Camera**: 1280x720 resolution, 60 FPS target
- **Detection**: MediaPipe Pose (33 landmarks) + Hands (21 landmarks per hand)
- **Physics**: Gravity, bounce factor, flap boost system
- **Performance**: Real-time processing with FPS display

## Troubleshooting

- **Camera not accessible**: Check macOS camera permissions in System Preferences
- **Low FPS**: Try reducing camera resolution or model complexity
- **Detection issues**: Ensure good lighting and clear view of upper body
