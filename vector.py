import math


class Vector:
    def __init__(self, xPos, yPos, nX, nY):
        self.xPos = xPos
        self.yPos = yPos
        self.nX = nX
        self.nY = nY

    def length(self):
        return math.dist((self.xPos, self.yPos), (self.nX, self.nY))
