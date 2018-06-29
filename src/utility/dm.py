import numpy as np

"""
    0 1 2 3 4
    ->
    3 0 4 1 2

    1. [0 1] [2 3]
    2. [3 0] [4 1]
"""


class Batch:
    def __init__(self, x, y, batch_size=1):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.index = 0

    def reachToLast(self):
        return self.x.shape[0] < self.index + self.batch_size


    def nextTo(self):
        if self.reachToLast():
            self.index = 0
            np.random.shuffle(self.x)
            np.random.shuffle(self.y)
        batch_x = self.x[self.index : self.index + self.batch_size]
        batch_y = self.y[self.index : self.index + self.batch_size]
        return batch_x, batch_y