import math
import random
import numpy as np
import matplotlib.pyplot as plt

DIMENSION = 11
Y_MAX = 3.0
DATA_N = 20


class LinerModel():
    def __init__(self, dim, λ, Λ, m):
        self.dim = dim
        self.λ = λ
        self.Λ = Λ
        self.m = m.reshape(self.dim, 1)

        self.Λ_hat = np.copy(self.Λ)
        self.m_hat = np.copy(self.m)

    def fit(self, xs, ys):
        num = len(ys)

        if len(xs) == 0:
            return

        if self.dim == 1:
            x = np.ones(xs.shape[0])
            self.Λ_hat = self.λ * np.sum(np.dot(x, np.transpose(x))) + self.Λ

            tmp = np.sum(x * ys)
            r_hat_inv = np.linalg.inv(self.Λ_hat)

            self.m_hat = r_hat_inv * (self.λ * tmp + np.dot(self.Λ, self.m))
        else:
            x_e = []
            for i in range(num):
                x_e.append(expand(xs[i], self.dim))
            x_e = np.array(x_e)

            sum_ = np.zeros((self.dim, self.dim))
            for i in range(x_e.shape[0]):
                sum_ += np.dot(x_e[i], x_e[i].T)

            self.Λ_hat = self.λ * sum_ + self.Λ

            Λ_hat_inv = np.linalg.inv(self.Λ_hat)

            sum_ = np.zeros((self.dim, 1))
            for i in range(num):
                sum_ += ys[i] * x_e[i]

            tmp = self.λ * sum_ + np.dot(self.Λ, self.m)
            self.m_hat = np.dot(Λ_hat_inv, tmp)

    def predict(self, x):
        if self.dim == 1:
            dev = math.sqrt(1/self.Λ_hat)
            return self.m_hat[0], dev
        else:
            x_e = expand(x, self.dim)
            mu = np.dot(self.m_hat.T, x_e)[0][0]

            Λ_hat_inv = np.linalg.inv(self.Λ_hat)
            tmp = np.dot(x_e.T, Λ_hat_inv)
            tmp2 = np.dot(tmp, x_e)

            lambda_star = (1 / self.λ) + tmp2
            dev = math.sqrt(lambda_star[0][0])
            return mu, dev

    def evidence(self):
        second_term = np.dot(np.dot(self.m.T, self.Λ), self.m)
        third_term = -np.log(np.linalg.det(self.Λ))
        forth_term = -np.dot(np.dot(self.m_hat.T, self.Λ_hat), self.m_hat)
        fifth_term = np.log(np.linalg.det(self.Λ_hat))

        evidence = - 0.5 * (second_term + third_term + forth_term + fifth_term)
        return evidence[0][0]


def expand(x, dim):
    val = []
    for d in range(dim):
        r = x ** d
        val.append(r)

    val_t = np.array(val).reshape((dim, 1))
    return val_t


def make_data(n=10):
    data = []
    for i in range(n):
        x = random.random() * 2 * math.pi
        y = math.sin(x) + 0.15
        data.append((x, y))

    return np.array(data)


def plot_data(data):
    xs = []
    ys = []
    for x, y in data:
        xs.append(x)
        ys.append(y)

    plt.scatter(xs, ys)
    plt.show()


def plot_prediction(data, model, fname=None):
    # データのプロット
    xs = []
    ys = []
    for x, y in data:
        xs.append(x)
        ys.append(y)

    plt.scatter(xs, ys)

    xs = []
    ys = []
    yuppers = []
    ylowers = []


    # 確率分布のプロット
    for x in np.arange(0, 2.5 * math.pi, 0.1):
        y, dev = model.predict(x)

        xs.append(x)
        ys.append(y)
        # dev = math.sqrt(1/model.Λ_hat)
        yuppers.append(y+dev)
        ylowers.append(y-dev)

    plt.plot(xs, ys)
    plt.plot(xs, yuppers, linestyle='dashed', color='blue')
    plt.plot(xs, ylowers, linestyle='dashed', color='blue')

    plt.ylim(-Y_MAX, Y_MAX)

    if fname is not None:
        plt.savefig(fname)
    plt.show()


def main():
    random.seed(0)
    data = make_data(DATA_N)

    for dim in range(1, DIMENSION):
        Λ = np.identity(dim)
        m = np.zeros(dim)
        λ = 1.0

        model = LinerModel(dim, λ, Λ, m)
        model.fit(data[:, 0], data[:, 1])
        evi = model.evidence()
        print(dim, evi)

        fname = "img/output%02d.png" % dim
        plot_prediction(data[:], model, fname)


if __name__ == '__main__':
    main()


