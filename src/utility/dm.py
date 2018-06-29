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
        self.iList = np.array([i for i in range(self.x.shape[0])])

    def reachToLast(self):
        return self.x.shape[0] < self.index + self.batch_size

    def nextTo(self):
        if self.reachToLast():
            self.index = 0
            np.random.shuffle(self.iList)
        batch_x = self.getBatch(self.x, self.iList[self.index : self.index + self.batch_size])
        batch_y = self.getBatch(self.y, self.iList[self.index : self.index + self.batch_size])
        return batch_x, batch_y