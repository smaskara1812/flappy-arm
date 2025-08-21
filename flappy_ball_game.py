import cv2
import numpy as np
import random
import time

class Game:
    """
    Encapsulates the entire Flappy Bird game logic and rendering.
    This class is independent of the camera and pose detection modules.
    """
    def __init__(self, screen_width, screen_height):
        # --- Screen and Asset Setup ---
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # --- Game Constants ---
        self.gravity = 0.4
        self.flap_strength = -8
        self.pipe_speed = 5
        self.pipe_gap = 200  # Vertical gap between pipes
        self.pipe_frequency = 1500  # Milliseconds
        
        # --- Bird Properties ---
        self.bird_start_pos = (screen_width // 4, screen_height // 2)
        self.bird_radius = 20
        self.bird_color = (0, 255, 255) # Yellow
        self.bird_outline_color = (0, 0, 0)

        # --- Pipe Properties ---
        self.pipe_width = 80
        self.pipe_color = (0, 255, 0) # Green
        self.pipe_outline_color = (0, 0, 0)

        # --- Background/Ground ---
        self.bg_color = (204, 153, 102) # A sky-blueish color
        self.ground_height = 50
        self.ground_y = self.screen_height - self.ground_height
        self.ground_color = (51, 153, 255) # A brownish color
        
        # --- Game State ---
        self.game_state = 'START' # Can be 'START', 'PLAYING', 'GAME_OVER'
        self.last_pipe_time = 0
        self.score = 0
        self.high_score = 0
        self.reset_game()

    def reset_game(self):
        """Resets the game to its initial state."""
        self.bird_pos = list(self.bird_start_pos)
        self.bird_velocity = 0
        self.pipes = []
        self.score = 0
        self.game_state = 'START'
        print("Game reset. Flap to start!")

    def flap(self):
        """Applies an upward velocity to the bird, initiating a flap."""
        if self.game_state == 'PLAYING':
            self.bird_velocity = self.flap_strength
        elif self.game_state == 'START':
            self.game_state = 'PLAYING'
            self.bird_velocity = self.flap_strength
            # Start spawning pipes after the first flap
            self.last_pipe_time = time.time() * 1000

    def _update_bird(self):
        """Updates the bird's position based on gravity and velocity."""
        if self.game_state == 'PLAYING':
            self.bird_velocity += self.gravity
            self.bird_pos[1] += self.bird_velocity

    def _create_pipe(self):
        """Creates a new pair of pipes with a random gap position."""
        gap_center = random.randint(self.pipe_gap, self.screen_height - self.pipe_gap)
        top_pipe_end = gap_center - self.pipe_gap // 2
        bottom_pipe_start = gap_center + self.pipe_gap // 2

        # [x, y, width, height, scored]
        top_pipe = [self.screen_width, 0, self.pipe_width, top_pipe_end, False]
        bottom_pipe = [self.screen_width, bottom_pipe_start, self.pipe_width, self.screen_height, False]
        self.pipes.extend([top_pipe, bottom_pipe])

    def _update_pipes(self):
        """Moves pipes to the left, removes old ones, and adds new ones."""
        if self.game_state != 'PLAYING':
            return

        # Move existing pipes
        for pipe in self.pipes:
            pipe[0] -= self.pipe_speed

        # Check for score
        bird_center_x = self.bird_pos[0]
        for pipe in self.pipes:
            if not pipe[4] and pipe[0] + self.pipe_width < bird_center_x:
                pipe[4] = True
                self.score += 0.5 # Each pair of pipes gives 1 point
        
        # Remove pipes that are off-screen
        self.pipes = [p for p in self.pipes if p[0] > -self.pipe_width]

        # Add new pipes periodically
        current_time = time.time() * 1000
        if current_time - self.last_pipe_time > self.pipe_frequency:
            self._create_pipe()
            self.last_pipe_time = current_time

    def _check_collisions(self):
        """Checks for collisions between the bird and pipes/boundaries."""
        # Unpack bird position for clarity
        bx, by = self.bird_pos
        br = self.bird_radius

        # Check collision with ground and ceiling
        if by + br > self.ground_y or by - br < 0:
            return True

        # Check collision with pipes
        for pipe in self.pipes:
            px, py, pw, ph = pipe[:4]
            # Simple Axis-Aligned Bounding Box (AABB) collision detection
            if (bx + br > px and bx - br < px + pw and
                by + br > py and by - br < py + ph):
                return True
        return False

    def _draw_background(self, frame):
        """Draws the static background and ground."""
        frame[:] = self.bg_color
        # Ground
        cv2.rectangle(frame, (0, self.ground_y), (self.screen_width, self.screen_height), self.ground_color, -1)
        # Ground outline
        cv2.line(frame, (0, self.ground_y), (self.screen_width, self.ground_y), (0,0,0), 2)


    def _draw_bird(self, frame):
        """Draws the bird."""
        pos = (int(self.bird_pos[0]), int(self.bird_pos[1]))
        cv2.circle(frame, pos, self.bird_radius, self.bird_color, -1)
        cv2.circle(frame, pos, self.bird_radius, self.bird_outline_color, 2)

    def _draw_pipes(self, frame):
        """Draws all the pipes."""
        for pipe in self.pipes:
            px, py, pw, ph = pipe[:4]
            cv2.rectangle(frame, (int(px), int(py)), (int(px + pw), int(py + ph)), self.pipe_color, -1)
            cv2.rectangle(frame, (int(px), int(py)), (int(px + pw), int(py + ph)), self.pipe_outline_color, 2)

    def _draw_ui(self, frame):
        """Draws the score and game state messages."""
        # Score
        score_text = f"Score: {int(self.score)}"
        cv2.putText(frame, score_text, (10, 50), self.font, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, score_text, (10, 50), self.font, 1.5, (0, 0, 0), 2)
        
        if self.game_state == 'START':
            msg = "Flap Arms to Start"
            text_size, _ = cv2.getTextSize(msg, self.font, 1.5, 3)
            text_x = (self.screen_width - text_size[0]) // 2
            cv2.putText(frame, msg, (text_x, self.screen_height // 2), self.font, 1.5, (255, 255, 255), 3)

        elif self.game_state == 'GAME_OVER':
            # Game Over message
            msg = "GAME OVER"
            text_size, _ = cv2.getTextSize(msg, self.font, 2, 4)
            text_x = (self.screen_width - text_size[0]) // 2
            cv2.putText(frame, msg, (text_x, self.screen_height // 2 - 50), self.font, 2, (255, 255, 255), 5)
            
            # Final Score message
            score_msg = f"Score: {int(self.score)} | High Score: {int(self.high_score)}"
            text_size, _ = cv2.getTextSize(score_msg, self.font, 1, 3)
            text_x = (self.screen_width - text_size[0]) // 2
            cv2.putText(frame, score_msg, (text_x, self.screen_height // 2 + 20), self.font, 1, (255, 255, 255), 3)
            
            # Reset message
            reset_msg = "Press 'r' to Restart"
            text_size, _ = cv2.getTextSize(reset_msg, self.font, 0.8, 2)
            text_x = (self.screen_width - text_size[0]) // 2
            cv2.putText(frame, reset_msg, (text_x, self.screen_height // 2 + 70), self.font, 0.8, (255, 255, 255), 2)

    def run_frame(self, frame):
        """
        Runs one frame of the game. Updates logic and draws all elements.
        
        Args:
            frame: The camera image (numpy array) to draw on.

        Returns:
            The frame with all game elements drawn on it.
        """
        # 1. Update game logic
        self._update_bird()
        self._update_pipes()
        if self.game_state == 'PLAYING' and self._check_collisions():
            self.game_state = 'GAME_OVER'
            if self.score > self.high_score:
                self.high_score = self.score
            print(f"Game Over! Final Score: {int(self.score)}")
            
        # 2. Draw all elements
        self._draw_background(frame)
        self._draw_pipes(frame)
        self._draw_bird(frame)
        self._draw_ui(frame)
        
        return frame