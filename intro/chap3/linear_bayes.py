import math
import random
import numpy as np
import matplotlib.pyplot as plt

DIMMENSION = 1
Y_MAX = 1.2


class LinerModel():
    def __init__(self, dim, λ, Λ, m):
        self.dim = dim
        self.λ = λ
        self.Λ = Λ
        self.m = m

        self.Λ_hat = self.Λ
        self.m_hat = m

    def fit(self, xs, ys):
        x = np.ones(xs.shape[0])
        self.Λ_hat = self.λ * np.sum(np.dot(x, np.transpose(x))) + self.Λ
        y = ys

        tmp = np.sum(x * y)
        r_hat_inv = np.linalg.inv(self.Λ_hat)

        self.m_hat = r_hat_inv * (self.λ * tmp + np.dot(self.Λ, self.m))


    def predict(self, x):
        None


def make_data(n=10):
    data = []
    for i in range(n):
        x = random.random() * 2 * math.pi
        y = math.sin(x) + 0.1
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


def predict(x, m_ast, lambda_s, RAMBDA):
    x_ast = 1
    u_ast = m_ast * x_ast
    lamda_ast_inv = (1/lambda_s) + (1 / RAMBDA)
    return u_ast, lamda_ast_inv


def plot_model(data, r_hat, m_hat):
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

    # 確率分布のプロット
    for x in np.arange(0, 2 * math.pi, 0.1):
        y = m_hat[0]

        xs.append(x)
        ys.append(y)

    plt.plot(xs, ys)
    plt.ylim(-Y_MAX, Y_MAX)

    plt.show()


def calc_r_hat(data, lambda_s, Rambda):

    x = np.ones(data.shape[0])
    r_hat = lambda_s * np.sum(np.dot(x, np.transpose(x))) + Rambda

    return r_hat


def calc_m_hat(data, r_hat, lambda_s, Rambda, m):
#    x = data[:,0]
    x = np.ones(data.shape[0])
    y = data[:,1]

    tmp = np.sum(x * y)
    r_hat_inv = np.linalg.inv(r_hat)

    m_hat = r_hat_inv * ( lambda_s *  tmp + np.dot(Rambda, m ))
    return m_hat




def main():
    random.seed(0)
    data = make_data(10)
    #plot_data(data)

    Λ = np.identity(DIMMENSION)
    m = np.zeros(DIMMENSION)
    λ = 1.0
    N =7

    λ_hat = calc_r_hat(data[:N], λ, Λ)
    m_hat = calc_m_hat(data[:N], λ_hat, λ, Λ, m)

    plot_model(data[:N], λ_hat, m_hat)

    print(λ_hat, m_hat)

    model = LinerModel(1, λ, Λ, m)
    model.fit(data[:N, 0], data[:N, 1])

    print(λ_hat, m_hat)


if __name__ == '__main__':
    main()


