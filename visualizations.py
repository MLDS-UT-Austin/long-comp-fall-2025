import math
import os
from datetime import datetime

import cv2
import pygame
from pygame.locals import *

# This file is for internal use to render games ####################################


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Load in images of players and spy
SPY_IMG = pygame.image.load("media/spy.png")
SPY_IMG = pygame.transform.scale(SPY_IMG, (100, 100))
PLAYER_IMG = pygame.image.load("media/player.jpg")
PLAYER_IMG = pygame.transform.scale(PLAYER_IMG, (100, 100))


class Visualization:
    def __init__(self, player_names: list[str], spy_indices: list[int], location: str):
        self.player_names = player_names
        self.spy_indices = spy_indices
        self.location = location

        # Initialize pygame
        pygame.init()

        # Screen dimensions
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

        # Font
        self.font = pygame.font.Font(
            None, 36
        )  # None for default font, 36 for font size

        # Initialize video recording
        self._setup_video_recording()

    def __del__(self):
        self._stop_recording()
        pygame.quit()

    def _setup_video_recording(self):
        """Initialize video recording with timestamped filename."""
        # Create recordings directory if it doesn't exist
        os.makedirs("recordings", exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_filename = os.path.join("recordings", f"game_{timestamp}.mp4")

        # Get screen dimensions
        width, height = self.screen.get_size()

        # Initialize video writer (using mp4v codec for MP4 format)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            self.video_filename,
            fourcc,
            30.0,  # 30 FPS for smooth video
            (width, height),
        )

        self.recording_active = True
        self.frame_count = 0

        print(f"Recording started: {self.video_filename}")

    def _stop_recording(self):
        """Stop video recording and save the file."""
        if hasattr(self, "video_writer") and self.video_writer is not None:
            self.recording_active = False
            self.video_writer.release()
            print(f"Recording saved: {self.video_filename}")
            self.video_writer = None

    def stop_recording(self):
        """Public method to manually stop recording before object destruction."""
        self._stop_recording()

    def wait_and_record(self, duration_ms):
        """Wait for a duration while continuously recording the current frame.

        This should be used instead of pygame.time.wait() to ensure frames are
        captured during wait periods.

        Args:
            duration_ms: Duration to wait in milliseconds
        """
        if self.recording_active and hasattr(self, "video_writer"):
            # Capture the current frame for the entire duration
            self._capture_frame(duration_ms=duration_ms)
        # Still perform the actual wait
        pygame.time.wait(duration_ms)

    def _capture_frame(self, duration_ms=0):
        """Capture the current pygame screen as a video frame.

        Args:
            duration_ms: How long this frame should be displayed (in milliseconds).
                        If > 0, the frame will be written multiple times to match the duration.
        """
        if (
            hasattr(self, "video_writer")
            and self.video_writer is not None
            and self.recording_active
        ):
            # Get the pygame surface as a string buffer
            frame = pygame.surfarray.array3d(self.screen)
            # Convert from (width, height, 3) to (height, width, 3)
            frame = frame.transpose([1, 0, 2])
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Calculate how many times to write this frame based on duration
            if duration_ms > 0:
                # At 30 FPS, each frame is ~33ms. Write frame multiple times for the duration.
                num_frames = max(1, int(duration_ms / 33.33))
            else:
                num_frames = 1

            for _ in range(num_frames):
                self.video_writer.write(frame)
                self.frame_count += 1

    def render_text(self, player_idx: int, msg: str):
        self.screen.fill(WHITE)
        text_surface = self.font.render(f"Location: {self.location}", True, BLACK)
        self.screen.blit(text_surface, pygame.Vector2(20, 20))

        # Display spy and player images on self.screen
        for i in range(len(self.player_names)):
            img = SPY_IMG if i in self.spy_indices else PLAYER_IMG
            self.screen.blit(img, self._player_pos(i) + pygame.Vector2(-50, -50))

        # Display player labels
        for i, player_name in enumerate(self.player_names):
            self._render_wrapped_text(player_name, self._label_pos(i))

        # Display the current dialogue
        self._render_wrapped_text(msg, self._text_pos(player_idx))

        # flip() the display to put your work on self.screen
        pygame.display.flip()

    # Function to wrap text
    def _wrap_text(self, text, font, max_width):
        words = text.split(" ")
        lines = []
        current_line = ""

        for word in words:
            # Check if adding the next word exceeds the max width
            test_line = f"{current_line} {word}".strip()
            if font.size(test_line)[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word  # Start a new line with the current word
        lines.append(current_line)  # Add the last line
        return lines

    # Function to render wrapped text using Vector2 for position
    def _render_wrapped_text(self, text, pos, color=BLACK, line_spacing=5):
        """Render text wrapped to fit within a specific width using a Vector2 for position."""
        lines = self._wrap_text(text, self.font, 0.3 * self._screen_width)

        if len(lines) > 7:
            lines = lines[:7]
            lines[-1] = lines[-1] + " ..."

        for i, line in enumerate(lines):
            text_surface = self.font.render(line, True, color)
            line_position = pos + pygame.Vector2(
                -self.font.size(line)[0] / 2,
                i * (self.font.get_linesize() + line_spacing),
            )
            self.screen.blit(text_surface, line_position)

    @property
    def _screen_width(self):
        return self.screen.get_size()[0]

    @property
    def _screen_height(self):
        return self.screen.get_size()[1]

    def _pos(self, player_idx: int, offset: int):
        cols = 3
        rows = max(1, math.ceil(len(self.player_names) / cols))
        row = player_idx // cols
        col = player_idx % cols

        # Clamp to ensure we never go out of the calculated grid
        row = min(row, rows - 1)
        col = min(col, cols - 1)

        x = self._screen_width * (col + 0.5) / cols
        y = self._screen_height * (0.8 * row + 0.5) / rows + offset
        return pygame.Vector2(x, y)

    def _player_pos(self, player_idx: int):
        return self._pos(player_idx, -65)

    def _label_pos(self, player_idx: int):
        return self._pos(player_idx, -150)

    def _text_pos(self, player_idx: int):
        return self._pos(player_idx, 0)


if __name__ == "__main__":
    vis = Visualization(
        [
            "Player 1",
            "Player 2",
            "Player 3",
            "Player 4",
            "Player 5",
            "Player 6",
        ],
        [1, 4],
        "Blanton Museum",
    )

    # Run a short test (3 rounds, ~18 seconds total)
    for round_num in range(3):
        for i in range(6):
            vis.render_text(
                i,
                f"Round {round_num + 1}: Player {i + 1} is speaking. This is a test message.",
            )
            vis.wait_and_record(1500)  # Wait 1.5 seconds and record

    # Clean up and save video
    del vis
    print("\nTest complete! Check the 'recordings' folder for the saved video.")
