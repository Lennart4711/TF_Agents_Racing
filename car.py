import math
import pygame
import numpy as np


class Car:
    def __init__(self, x, y, r, lasers):
        self.xPos = x
        self.yPos = y
        self.angle = r
        self.vX = 0
        self.vY = 0
        self.lasers = [np.zeros((4,))] * lasers
        self.driven_distance = 0

    # Sets the direction for every laser relative to the car's rotation
    def set_lasers(self):
        for i in range(len(self.lasers)):
            turn = 360 / len(self.lasers)
            turn = 20
            radians = (
                (270 + self.angle + i * turn - turn / 2 * (len(self.lasers) - 1))
                * math.pi
                / 180
            )
            x = self.xPos + 600 * math.cos(radians)
            y = self.yPos + 600 * math.sin(radians)
            self.lasers[i] = np.array([[self.xPos, self.yPos], [x, y]])

    def draw(self, win):
        pygame.draw.circle(win, (211, 123, 23), (self.xPos, self.yPos), 5)
        laserColor = (155, 20, 155)
        for x in self.lasers:
            pygame.draw.line(
                win, laserColor, (x[1][0], x[1][1]), (self.xPos, self.yPos)
            )

    def accelerate(self, b):
        xN = math.sin(2 * math.pi * (self.angle / 360))
        yN = math.cos(2 * math.pi * (self.angle / 360))
        # Adds to velocity vector, using minus for y because pygame uses 0,0 as top-left corner
        self.vX += xN * b * 0.05
        self.vY -= yN * b * 0.05

    def turn(self, direction):
        self.angle += direction  # between -1 and 1
        if self.angle > 720 or self.angle < -720:
            self.angle = 0

    def update(self):
        # Keep velocity within bounds
        self.vX = min(self.vX, 5)
        self.vX = max(self.vX, -5)
        self.vY = min(self.vY, 5)
        self.vY = max(self.vY, -5)

        self.set_lasers()
        self.xPos += self.vX
        self.yPos += self.vY
        self.driven_distance = math.sqrt(self.vX**2 + self.vY**2)
        # drag force
        self.vX *= 0.99
        self.vY *= 0.99

    def move(self, steer, acc):
        self.turn(steer)
        self.accelerate(acc)

    def crashed(self) -> bool:
        return any(
            math.dist((self.xPos, self.yPos), (laser[1][0], laser[1][1])) < 5
            for laser in self.lasers
        )

    def set_laser_length(self, borders):
        for x in self.lasers:
            for y in borders:
                v = self.get_collision_point(x, y)
                if v is not None:
                    x[1][0] = v[0]
                    x[1][1] = v[1]
            d = math.dist((self.xPos, self.yPos), (x[1][0], x[1][1]))

    def get_collision_point(self, a, b):
        x1 = a[0][0]
        x2 = a[1][0]
        y1 = a[0][1]
        y2 = a[1][1]
        x3 = b[0][0]
        x4 = b[1][0]
        y3 = b[0][1]
        y4 = b[1][1]

        denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denominator == 0:
            # Line segments are parallel
            return None

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

        # Do the lines intersect in the given segments?
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            intersectionX = x1 + (ua * (x2 - x1))
            intersectionY = y1 + (ua * (y2 - y1))
            return [intersectionX, intersectionY]
        else:
            return None
