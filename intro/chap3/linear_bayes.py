import math
import random
import numpy as np
import matplotlib.pyplot as plt

DIMMENSION = 5
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


def predict(x, m_ast, lambda_s, RAMBDA):
    x_ast = 1
    u_ast = m_ast * x_ast
    lamda_ast_inv = (1/lambda_s) + (1 / RAMBDA)
    return u_ast, lamda_ast_inv


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
    for x in np.arange(0, 2 * math.pi, 0.1):
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
    data = make_data(DATA_N)
    #plot_data(data)

    Λ = np.identity(DIMMENSION)
    m = np.zeros(DIMMENSION)
    λ = 1.0

#    data = np.array([[2, 3], [0, 1]])
#    model = LinerModel(DIMMENSION, λ, Λ, m)
#    model.fit(data[:, 0], data[:, 1])
#    fname = "img/2dfit.png"
#   plot_prediction(data, model, fname)


    for N in range(DATA_N):
        model = LinerModel(DIMMENSION, λ, Λ, m)
        model.fit(data[:N, 0], data[:N, 1])

        print(model.Λ_hat, model.m_hat)
        fname = "img/output%02d.png" % N
        plot_prediction(data[:N], model, fname)


if __name__ == '__main__':
    main()


