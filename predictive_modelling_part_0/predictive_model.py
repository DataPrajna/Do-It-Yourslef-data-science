import tensorflow as tf
from abc import ABCMeta, abstractmethod

class Model(ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def cost_function(self):
        pass

    @abstractmethod
    def solver(self):
        pass

    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def predict(self):
        pass


