import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

pygame.init()


class Editor:
    def __init__(self):
        self.run = True
        self.win = pygame.display.set_mode((600, 600))
        pygame.display.set_caption("Racing game")
        self.clock = pygame.time.Clock()
        self.borders = []
        self.active_border = Border()
        self.borders.append(self.active_border)

    def input(self):
        """Every click adds the starting or end point to a border"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # if last border has started and not ended
                    if (
                        self.active_border.has_started()
                        and not self.active_border.has_ended()
                    ):
                        # set end_pos to mouse position
                        self.active_border.set_end_pos(pygame.mouse.get_pos())
                        # create new border
                        self.active_border = Border()
                        self.borders.append(self.active_border)
                        self.active_border.set_start_pos(pygame.mouse.get_pos())
                    # if last border has not started
                    elif not self.active_border.has_started():
                        # set start_pos to mouse position
                        self.active_border.set_start_pos(pygame.mouse.get_pos())
                        # set has_started to True

                    # if last border has ended
                    elif self.active_border.has_ended():
                        # set start_pos to mouse position
                        self.active_border.set_start_pos(pygame.mouse.get_pos())

    def loop(self):
        while self.run:
            self.clock.tick(60)
            self.win.fill((0, 0, 0))
            self.input()
            for border in self.borders:
                border.draw(self.win)
            pygame.display.update()

        self.store_borders()

    def store_borders(self):
        """Store the borders as a list of position tuples in a file"""
        with open("borders.txt", "w") as file:
            out = []
            for border in self.borders:
                # Append a tupel for each border
                out.append((border.start_pos, border.end_pos))
            # Write the list to the file
            file.write(str(out))


class Border:
    def __init__(self):
        self.start_pos = None
        self.end_pos = None

    def set_start_pos(self, pos):
        self.start_pos = pos

    def set_end_pos(self, pos):
        self.end_pos = pos

    def has_started(self):
        return self.start_pos != None

    def has_ended(self):
        return self.end_pos != None

    def draw(self, window):
        # Draw line
        if self.start_pos and self.end_pos:
            pygame.draw.line(window, (255, 255, 255), self.start_pos, self.end_pos, 2)


if __name__ == "__main__":
    editor = Editor()
    editor.loop()
