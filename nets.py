import numpy as np
from scipy.stats import logistic
from scipy.special import softmax


sigmoid = logistic.cdf


def sigmoid_derivative(x):
    y = sigmoid(x)
    return (1 - y) * y


class Elman:
    def __init__(self, input_n, hidden_n, output_n):
        self.input_n = input_n
        self.hidden_n = hidden_n
        self.output_n = output_n

        self.layers = []

        # определение вектора с нейронами для входного слоя, +1 нейрон на смещение и контекстный слой(длина скрытого)
        self.layers.append(np.zeros(input_n + 1 + hidden_n))

        # скрытый слой
        self.layers.append(np.zeros(hidden_n))

        # выходной слой
        self.layers.append(np.zeros(output_n))

        # массив весов
        self.weights = []

        for i in range(2):
            # сделаем инициализацию Ксавьера https://pytorch.org/docs/stable/nn.init.html?highlight=xavier#torch.nn.init.xavier_uniform_
            bound = (6/(self.layers[i].size + self.layers[i+1].size))**(1/2)
            self.weights.append(np.random.uniform(-bound, bound, (self.layers[i].size, self.layers[i + 1].size)))

        # для метода моментов сохраняем предыдущие производные
        self.dw = [0, ] * len(self.weights)

    def forward(self, x):
        # инициализируем входной слой постувшими данными
        self.layers[0][:self.input_n] = x

        # инициализация контекстного слоя
        self.layers[0][self.input_n:-1] = self.layers[1]

        for i in range(1, 3):
            self.layers[i][...] = sigmoid(np.dot(self.layers[i - 1], self.weights[i - 1]))

        return self.layers[-1], softmax(self.layers[-1])

    def backward(self, target, lrate=0.1, momentum=0.1):
        deltas = []

        # подсчет ошибок на выходном слое (используется производная от кросс-энропии с софтмаксом)
        error = target - self.layers[-1]
        delta = error * sigmoid_derivative(self.layers[-1])
        deltas.append(delta)

        # Ошибка на скрытых слоях
        for i in range(1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * sigmoid_derivative(self.layers[i])
            deltas.insert(0, delta)

        # обновляем веса
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            self.weights[i] += lrate * dw + momentum * self.dw[i]
            self.dw[i] = dw


if __name__ == '__main__':
    network = Elman(4, 3, 4)
