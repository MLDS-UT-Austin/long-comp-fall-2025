import math
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import cv2
import pygame

# This file is for internal use to render games ####################################


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Load in images of players and spy
SPY_IMG = pygame.image.load("media/spy.png")
SPY_IMG = pygame.transform.scale(SPY_IMG, (100, 100))
PLAYER_IMG = pygame.image.load("media/player.jpg")
PLAYER_IMG = pygame.transform.scale(PLAYER_IMG, (100, 100))


class Visualization:
    def __init__(
        self,
        player_names: list[str],
        spy_indices: list[int],
        location: str,
        *,
        output_path: str,
        resolution: tuple[int, int] = (1920, 1080),
        fps: float = 30.0,
    ):
        self.player_names = player_names
        self.spy_indices = spy_indices
        self.location = location
        self.resolution = resolution
        self.fps = fps

        pygame.init()
        pygame.font.init()
        self.surface = pygame.Surface(self.resolution)

        # Font
        self.font = pygame.font.Font(
            None, 36
        )  # None for default font, 36 for font size

        # Initialize video recording
        self.video_writer: cv2.VideoWriter | None = None
        self.video_path: str | None = None
        self.frame_count = 0
        self.recording_active = False
        self._setup_video_recording(output_path)

    def __del__(self):
        self.close()

    def close(self):
        self._stop_recording()
        pygame.quit()

    def _setup_video_recording(self, output_path: str):
        """Initialize video recording to write directly to the provided path."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Initialize video writer (using mp4v codec for MP4 format)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        self.video_writer = cv2.VideoWriter(
            str(target),
            fourcc,
            self.fps,
            self.resolution,
        )

        self.video_path = str(target)
        self.recording_active = True

    def _stop_recording(self):
        """Stop video recording and save the file."""
        if hasattr(self, "video_writer") and self.video_writer is not None:
            self.recording_active = False
            self.video_writer.release()
            self.video_writer = None
            self.video_path = None

    def stop_recording(self):
        """Public method to manually stop recording before object destruction."""
        self._stop_recording()

    def record_duration(self, duration_ms: int):
        """Record the current frame for the requested duration without waiting in real time."""
        if duration_ms <= 0:
            duration_ms = int(1000 / self.fps)
        self._capture_frame(duration_ms=duration_ms)

    def _capture_frame(self, duration_ms=0):
        """Capture the current pygame surface as a video frame.

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
            frame = pygame.surfarray.array3d(self.surface)
            # Convert from (width, height, 3) to (height, width, 3)
            frame = frame.transpose([1, 0, 2])
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Calculate how many times to write this frame based on duration
            if duration_ms > 0:
                frame_duration_ms = 1000 / self.fps
                num_frames = max(1, int(math.ceil(duration_ms / frame_duration_ms)))
            else:
                num_frames = 1

            for _ in range(num_frames):
                self.video_writer.write(frame)
                self.frame_count += 1

    def render_text(self, player_idx: int, msg: str):
        self.surface.fill(WHITE)
        text_surface = self.font.render(f"Location: {self.location}", True, BLACK)
        self.surface.blit(text_surface, pygame.Vector2(20, 20))

        # Display spy and player images on self.screen
        for i in range(len(self.player_names)):
            img = SPY_IMG if i in self.spy_indices else PLAYER_IMG
            self.surface.blit(img, self._player_pos(i) + pygame.Vector2(-50, -50))

        # Display player labels
        for i, player_name in enumerate(self.player_names):
            self._render_wrapped_text(player_name, self._label_pos(i))

        # Display the current dialogue
        self._render_wrapped_text(msg, self._text_pos(player_idx))

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
            self.surface.blit(text_surface, line_position)

    @property
    def _screen_width(self):
        return self.surface.get_size()[0]

    @property
    def _screen_height(self):
        return self.surface.get_size()[1]

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
