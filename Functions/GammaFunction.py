import numpy as np

class GammaFunction:
    def __init__(self, gamma=0.10):
        self.gamma = gamma

    def __call__(self, x):
        return int(np.floor(self.gamma * x))  # without upper bound for the number of lower samples

