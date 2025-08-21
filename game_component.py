import cv2
import numpy as np
import random
import time

class GameComponent:
    def __init__(self, game_width, game_height):
        self.game_width = game_width
        self.game_height = game_height
        
        # --- Screen and Asset Setup ---
        self.screen_width = game_width
        self.screen_height = game_height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # --- Game Constants ---
        self.gravity = 0.4
        self.flap_strength = -8
        self.pipe_speed = 5  # Slower pipe movement
        
        # --- Dynamic Pipe Settings ---
        self.base_pipe_gap = 200  # Base vertical gap between pipes
        self.base_pipe_frequency = 7000  # Base time between pipes (7 seconds)
        self.min_pipe_gap = 180  # Minimum gap (hardest)
        self.max_pipe_gap = 220  # Maximum gap (easiest)
        self.min_pipe_frequency = 4000  # Minimum time between pipes (3 seconds)
        self.max_pipe_frequency = 6000  # Maximum time between pipes (8 seconds)
        
        # --- Bird Properties ---
        self.bird_start_pos = (game_width // 4, game_height // 2)
        self.bird_radius = 20
        self.bird_color = (0, 255, 255) # Yellow
        self.bird_outline_color = (0, 0, 0)
        self.bird_eye_color = (255, 255, 255) # White
        self.bird_beak_color = (255, 165, 0) # Orange
        
        # Bird animation properties
        self.bird_rotation = 0
        self.wing_angle = 0
        self.wing_speed = 0
        self.last_flap_time = 0
        self.flap_animation_duration = 200  # milliseconds

        # --- Pipe Properties ---
        self.pipe_width = 80
        self.pipe_color = (34, 139, 34) # Forest green (more retro)
        self.pipe_outline_color = (0, 100, 0) # Dark green outline
        self.pipe_highlight_color = (50, 205, 50) # Light green highlight
        self.pipe_cap_color = (0, 128, 0) # Darker green for pipe caps

        # --- Background/Ground ---
        self.bg_color = (135, 206, 235) # Sky blue background
        self.ground_height = 80
        self.ground_y = self.screen_height - self.ground_height
        self.ground_color = (34, 139, 34) # Forest green ground
        self.grass_color = (50, 205, 50) # Light green grass
        self.grass_dark_color = (0, 100, 0) # Dark green grass detail
        
        # Scrolling ground effect
        self.ground_scroll_x = 0
        self.ground_scroll_speed = 3
        
        # --- Game State ---
        self.game_state = 'START' # Can be 'START', 'PLAYING', 'GAME_OVER'
        self.last_pipe_time = 0
        self.score = 0
        self.high_score = 0
        self.last_flap_count = 0
        self.reset_game()

    def reset_game(self):
        """Resets the game to its initial state."""
        self.bird_pos = list(self.bird_start_pos)
        self.bird_velocity = 0
        self.pipes = []
        self.score = 0
        self.game_state = 'START'
        self.last_flap_count = 0
        print("Game reset. Flap to start!")

    def update_physics(self, flap_count):
        """Updates game physics based on flap count from camera detection."""
        # Check for new flaps
        if flap_count > self.last_flap_count:
            self.flap()
            self.last_flap_count = flap_count
        
        # Update game logic
        self._update_bird()
        self._update_pipes()
        if self.game_state == 'PLAYING' and self._check_collisions():
            self.game_state = 'GAME_OVER'
            if self.score > self.high_score:
                self.high_score = self.score
            print(f"Game Over! Final Score: {int(self.score)}")

    def flap(self):
        """Applies an upward velocity to the bird, initiating a flap."""
        if self.game_state == 'PLAYING':
            self.bird_velocity = self.flap_strength
        elif self.game_state == 'START':
            self.game_state = 'PLAYING'
            self.bird_velocity = self.flap_strength
            # Start spawning pipes after the first flap
            self.last_pipe_time = time.time() * 1000
        
        # Trigger wing animation
        self.wing_speed = 15
        self.last_flap_time = time.time() * 1000

    def _update_bird(self):
        """Updates the bird's position based on gravity and velocity."""
        if self.game_state == 'PLAYING':
            self.bird_velocity += self.gravity
            self.bird_pos[1] += self.bird_velocity
            
            # Update bird rotation based on velocity
            target_rotation = min(max(self.bird_velocity * 3, -45), 45)
            self.bird_rotation += (target_rotation - self.bird_rotation) * 0.1
            
            # Update wing animation
            self.wing_angle += self.wing_speed
            self.wing_speed *= 0.9  # Decay wing speed

    def _get_dynamic_pipe_gap(self):
        """Calculate dynamic pipe gap based on score."""
        # Start easy, get harder as score increases
        if self.score < 5:
            return self.max_pipe_gap  # Easiest
        elif self.score < 15:
            return self.max_pipe_gap - (self.score - 5) * 8  # Gradually decrease
        else:
            return self.min_pipe_gap  # Hardest
    
    def _get_dynamic_pipe_frequency(self):
        """Calculate dynamic pipe frequency based on score."""
        # Start with more time, get faster as score increases
        if self.score < 5:
            return self.max_pipe_frequency  # Most time between pipes
        elif self.score < 15:
            return self.max_pipe_frequency - (self.score - 5) * 300  # Gradually decrease
        else:
            return self.min_pipe_frequency  # Least time between pipes

    def _create_pipe(self):
        """Creates a new pair of pipes with a random gap position."""
        # Get dynamic values
        current_pipe_gap = self._get_dynamic_pipe_gap()
        
        gap_center = random.randint(current_pipe_gap, self.screen_height - current_pipe_gap)
        top_pipe_end = gap_center - current_pipe_gap // 2
        bottom_pipe_start = gap_center + current_pipe_gap // 2

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

        # Update ground scroll
        self.ground_scroll_x -= self.ground_scroll_speed
        if self.ground_scroll_x <= -50:  # Reset scroll position
            self.ground_scroll_x = 0

        # Check for score
        bird_center_x = self.bird_pos[0]
        for pipe in self.pipes:
            if not pipe[4] and pipe[0] + self.pipe_width < bird_center_x:
                pipe[4] = True
                self.score += 0.5 # Each pair of pipes gives 1 point
        
        # Remove pipes that are off-screen
        self.pipes = [p for p in self.pipes if p[0] > -self.pipe_width]

        # Add new pipes periodically with dynamic frequency
        current_time = time.time() * 1000
        current_pipe_frequency = self._get_dynamic_pipe_frequency()
        if current_time - self.last_pipe_time > current_pipe_frequency:
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
        """Draws the background with scrolling grass ground."""
        # Sky background
        frame[:] = self.bg_color
        
        # Main ground
        cv2.rectangle(frame, (0, self.ground_y), (self.screen_width, self.screen_height), self.ground_color, -1)
        
        # Scrolling grass detail
        grass_pattern_width = 50
        for x in range(-grass_pattern_width + int(self.ground_scroll_x), self.screen_width + grass_pattern_width, grass_pattern_width):
            # Draw grass tufts
            for i in range(0, self.screen_width, 30):
                grass_x = i + x
                if 0 <= grass_x < self.screen_width:
                    # Grass tuft
                    grass_y = self.ground_y - 5
                    cv2.line(frame, (grass_x, grass_y), (grass_x, grass_y - 15), self.grass_color, 2)
                    cv2.line(frame, (grass_x - 2, grass_y - 5), (grass_x + 2, grass_y - 10), self.grass_color, 1)
                    cv2.line(frame, (grass_x - 1, grass_y - 8), (grass_x + 1, grass_y - 12), self.grass_color, 1)
        
        # Ground texture lines
        for y in range(self.ground_y + 10, self.screen_height, 15):
            cv2.line(frame, (0, y), (self.screen_width, y), self.grass_dark_color, 1)
        
        # Ground outline
        cv2.line(frame, (0, self.ground_y), (self.screen_width, self.ground_y), (0, 0, 0), 2)

    def _draw_bird(self, frame):
        """Draws the bird in classic Flappy Bird style."""
        pos = (int(self.bird_pos[0]), int(self.bird_pos[1]))
        
        # Calculate bird rotation for realistic tilt
        rotation_angle = self.bird_rotation * np.pi / 180
        
        # Draw main bird body (oval shape for more realistic look)
        body_width = self.bird_radius * 2
        body_height = int(self.bird_radius * 1.6)
        
        # Create bird body as filled ellipse
        cv2.ellipse(frame, pos, (body_width//2, body_height//2), 
                   rotation_angle * 180 / np.pi, 0, 360, self.bird_color, -1)
        cv2.ellipse(frame, pos, (body_width//2, body_height//2), 
                   rotation_angle * 180 / np.pi, 0, 360, self.bird_outline_color, 2)
        
        # Draw bird head (slightly larger circle)
        head_radius = int(self.bird_radius * 0.8)
        head_pos = (pos[0] + int(8 * np.cos(rotation_angle)), 
                   pos[1] - int(5 * np.sin(rotation_angle)))
        cv2.circle(frame, head_pos, head_radius, self.bird_color, -1)
        cv2.circle(frame, head_pos, head_radius, self.bird_outline_color, 2)
        
        # Draw classic Flappy Bird eye (single large eye)
        eye_radius = 6
        eye_pos = (head_pos[0] + 3, head_pos[1] - 3)
        cv2.circle(frame, eye_pos, eye_radius, (255, 255, 255), -1)  # White eye
        cv2.circle(frame, eye_pos, eye_radius, self.bird_outline_color, 1)
        
        # Draw pupil
        pupil_pos = (eye_pos[0] + 1, eye_pos[1] - 1)
        cv2.circle(frame, pupil_pos, 3, (0, 0, 0), -1)
        
        # Draw classic beak (triangle shape)
        beak_points = np.array([
            [head_pos[0] + head_radius + 2, head_pos[1]],
            [head_pos[0] + head_radius + 12, head_pos[1] - 3],
            [head_pos[0] + head_radius + 12, head_pos[1] + 3]
        ], np.int32)
        cv2.fillPoly(frame, [beak_points], self.bird_beak_color)
        cv2.polylines(frame, [beak_points], True, self.bird_outline_color, 1)
        
        # Draw wing (classic Flappy Bird style)
        wing_offset = int(5 * np.sin(self.wing_angle * 0.1))
        wing_pos = (pos[0] - 8, pos[1] + wing_offset)
        wing_radius = int(self.bird_radius * 0.6)
        cv2.circle(frame, wing_pos, wing_radius, self.bird_color, -1)
        cv2.circle(frame, wing_pos, wing_radius, self.bird_outline_color, 1)
        
        # Draw wing detail
        wing_detail_pos = (wing_pos[0] - 3, wing_pos[1] - 2)
        cv2.circle(frame, wing_detail_pos, 4, (255, 255, 255), -1)
        cv2.circle(frame, wing_detail_pos, 4, self.bird_outline_color, 1)

    def _draw_pipes(self, frame):
        """Draws all the pipes in retro Flappy Bird style."""
        for pipe in self.pipes:
            px, py, pw, ph = pipe[:4]
            px, py, pw, ph = int(px), int(py), int(pw), int(ph)
            
            # Draw main pipe body
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), self.pipe_color, -1)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), self.pipe_outline_color, 2)
            
            # Draw pipe cap (wider top/bottom section)
            cap_height = 20
            cap_width = pw + 10
            
            # Top cap (for bottom pipes)
            if py > 0:  # If pipe starts below top of screen
                cap_x = px - 5
                cap_y = py - cap_height
                cv2.rectangle(frame, (cap_x, cap_y), (cap_x + cap_width, cap_y + cap_height), 
                             self.pipe_cap_color, -1)
                cv2.rectangle(frame, (cap_x, cap_y), (cap_x + cap_width, cap_y + cap_height), 
                             self.pipe_outline_color, 2)
            
            # Bottom cap (for top pipes)
            if py + ph < self.screen_height:  # If pipe ends above bottom of screen
                cap_x = px - 5
                cap_y = py + ph
                cv2.rectangle(frame, (cap_x, cap_y), (cap_x + cap_width, cap_y + cap_height), 
                             self.pipe_cap_color, -1)
                cv2.rectangle(frame, (cap_x, cap_y), (cap_x + cap_width, cap_y + cap_height), 
                             self.pipe_outline_color, 2)
            
            # Draw pipe highlights (retro shading effect)
            highlight_width = 15
            highlight_x = px + 5
            cv2.rectangle(frame, (highlight_x, py + 5), (highlight_x + highlight_width, py + ph - 5), 
                         self.pipe_highlight_color, -1)
            
            # Draw pipe details (horizontal lines for texture)
            detail_spacing = 30
            for i in range(0, ph, detail_spacing):
                if py + i < self.screen_height:
                    line_y = py + i
                    cv2.line(frame, (px, line_y), (px + pw, line_y), self.pipe_outline_color, 1)

    def _draw_ui(self, frame):
        """Draws the score and game state messages."""
        # Score with enhanced styling
        score_text = f"Score: {int(self.score)}"
        cv2.putText(frame, score_text, (10, 50), self.font, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, score_text, (10, 50), self.font, 1.5, (0, 0, 0), 2)
        
        # High score
        high_score_text = f"High Score: {int(self.high_score)}"
        cv2.putText(frame, high_score_text, (10, 90), self.font, 1, (255, 255, 0), 2)
        
        # Difficulty indicator
        current_gap = self._get_dynamic_pipe_gap()
        current_freq = self._get_dynamic_pipe_frequency()
        difficulty_text = f"Gap: {current_gap}px | Freq: {current_freq/1000:.1f}s"
        cv2.putText(frame, difficulty_text, (10, 130), self.font, 0.6, (200, 200, 200), 1)
        
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

    def render(self, canvas, offset_x=0, offset_y=0):
        """Render the game on the provided canvas"""
        # Create game frame
        game_frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        # Draw all game elements
        self._draw_background(game_frame)
        self._draw_pipes(game_frame)
        self._draw_bird(game_frame)
        self._draw_ui(game_frame)
        
        # Place game frame on canvas
        canvas[offset_y:offset_y + self.screen_height, 
               offset_x:offset_x + self.screen_width] = game_frame
    
    def reset(self):
        """Reset the game state"""
        self.reset_game()
