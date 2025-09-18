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
    def __init__(self, player_names: list[str], spy_index: int, location: str):
        self.player_names = player_names
        self.spy_index = spy_index
        self.location = location

        # Initialize pygame
        pygame.init()

        # Screen dimensions
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

        # Font
        self.font = pygame.font.Font(
            None, 36
        )  # None for default font, 36 for font size

    def __del__(self):
        pygame.quit()

    def render_text(self, player_idx: int, msg: str):
        self.screen.fill(WHITE)
        text_surface = self.font.render(f"Location: {self.location}", True, BLACK)
        self.screen.blit(text_surface, pygame.Vector2(20, 20))

        # Display spy and player images on self.screen
        for i in range(4):
            img = SPY_IMG if i == self.spy_index else PLAYER_IMG
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
        lines = self._wrap_text(text, self.font, 0.4 * self._screen_width)

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
        top = player_idx % 2
        bottom = int(player_idx / 2)
        return pygame.Vector2(
            self._screen_width * (top * 2 + 1) / 4,
            self._screen_height * (bottom * 1.83 + 1) / 4 + offset,
        )

    def _player_pos(self, player_idx: int):
        return self._pos(player_idx, -65)

    def _label_pos(self, player_idx: int):
        return self._pos(player_idx, -150)

    def _text_pos(self, player_idx: int):
        return self._pos(player_idx, 0)


if __name__ == "__main__":
    vis = Visualization(["Player 1", "Player 2", "Player 3", "Player 4"], 0, "Beach")
    while True:
        for i in range(4):
            vis.render_text(i, "This is too long to fit on one line. " * 20)
            pygame.time.wait(2000)
