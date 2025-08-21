import cv2
import time
from game_component import GameComponent
from camera_component import CameraComponent
import numpy as np

class MainGame:
    def __init__(self):
        # Screen dimensions
        self.screen_width = 1280
        self.screen_height = 720
        
        # Layout configuration
        self.game_width = int(self.screen_width * 0.7)  # 70% for game
        self.camera_width = 320
        self.camera_height = 240
        self.camera_x = self.game_width + 20  # Position camera after game area
        self.camera_y = 20
        
        # Initialize components
        self.game = GameComponent(self.game_width, self.screen_height)
        self.camera = CameraComponent(self.camera_width, self.camera_height)
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0.0
        
    def run(self):
        """Main game loop"""
        print("Flappy Ball Game - Modular Version")
        print("Press 'r' to reset, 'q' to quit")
        
        while True:
            # Create main canvas
            canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            
            # Get flap count from camera
            flap_count = self.camera.get_flap_count()
            
            # Update game physics
            self.game.update_physics(flap_count)
            
            # Render game component
            self.game.render(canvas, 0, 0)
            
            # Render camera component
            self.camera.render(canvas, self.camera_x, self.camera_y)
            
            # Calculate and display FPS
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.fps = self.fps_counter / (time.time() - self.fps_start_time)
                self.fps_counter = 0
                self.fps_start_time = time.time()
            
            # Display FPS
            cv2.putText(canvas, f"FPS: {self.fps:.1f}", (self.game_width + 20, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Flappy Ball Game - Modular', canvas)
            
            # Handle key presses and window close
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.game.reset()
                self.camera.reset_counters()
                print("Game reset!")
            
            # Check if window was closed
            if cv2.getWindowProperty('Flappy Ball Game - Modular', cv2.WND_PROP_VISIBLE) < 1:
                break
        
        # Cleanup
        self.camera.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = MainGame()
    game.run()
