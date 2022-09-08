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
            #self.lasers[i] = Vector(self.xPos, self.yPos, x, y)
            self.lasers[i] = np.array([[self.xPos, self.yPos], [x, y]])

    def draw_car(self, win):
        pygame.draw.circle(win, (211, 123, 23), (self.xPos, self.yPos), 5)
        laserColor = (155, 20, 155)
        for x in self.lasers:
            pygame.draw.line(win, laserColor, (x[1][0], x[1][1]), (self.xPos, self.yPos))

    def accelerate(self, b):
        xN = math.sin(2 * math.pi * (self.angle / 360))
        yN = math.cos(2 * math.pi * (self.angle / 360))
        # Adds to velocity vector, using minus for y because pygame uses 0,0 as top-left corner
        self.vX += xN * b * 0.05
        self.vY -= yN * b * 0.05

    def turn(self, direction):
        # direction = 1 or -1
        self.angle += 2 * direction
        if self.angle > 360:
            self.angle -= 360
        if self.angle < 0:
            self.angle += 360

    def update(self):
        self.set_lasers()

        self.xPos += self.vX
        self.yPos += self.vY
        # drag force on the velocity
        self.vX -= self.vX * 0.06
        self.vY -= self.vY * 0.06

    def move(self, steer, acc):
        self.turn(steer)
        self.accelerate(acc)

    def crashed(self) -> bool:
        for laser in self.lasers:
            if math.dist((self.xPos, self.yPos), (laser[1][0], laser[1][1])) < 5:
                return True
        return False
