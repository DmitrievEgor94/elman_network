from abc import ABCMeta, abstractmethod

import numpy as np


# абстрактный класс, который наследуют все модули и нейронные сети в этой библиотеке для указания методов реализации
class Module:
    __metaclass__ = ABCMeta

    # метод с описанием того, как преобразуются входные данные
    @abstractmethod
    def forward(self, x):
        pass

    # метод с реализацией градиентого спуска для этого слоя
    def backward(self, y):
        pass

    @abstractmethod
    def initialise(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(args[0])


# реализация линейного слоя
class Linear(Module):
    def __init__(self, input_n, output_n, bias=True):
        self.need_bias = bias
        self.bias = 0

        self.weights = np.zeros((input_n, output_n))
        self.last_inputs = np.zeros(input_n)
        self.gradients = np.zeros((input_n, output_n))

    def forward(self, x):
        self.last_inputs = x
        return np.dot(self.weights, x) + self.bias

    def backward(self, y):
        

    def initialise(self):
        np.random.uniform()


if __name__ == '__main__':
    lin_example = Linear(5, 6)

    print(lin_example(np.ones((6, 1))))
