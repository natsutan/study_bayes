import math
import random
import numpy as np
import matplotlib.pyplot as plt

DIMMENSION = 3
Y_MAX = 1.1


def make_data(n=10):
    data = []
    for i in range(n):
        x = random.random() * 2 * math.pi
        y = math.sin(x)
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
        y = m_hat[0] + m_hat[1] * x + m_hat[2] * (x**2)

        xs.append(x)
        ys.append(y)

    plt.plot(xs, ys)
    plt.ylim(-Y_MAX, Y_MAX)

    plt.show()

def calc_r_hat(data, lambda_s, Rambda):
    x = data[:,0]
    r_hat = lambda_s * np.sum(np.dot(x, np.transpose(x))) + Rambda

    return r_hat

def calc_m_hat(data, r_hat, lambda_s, Rambda, m):
    x = data[:,0]
    y = data[:,1]

    tmp = np.sum(x * y)

    m_hat = np.transpose(r_hat) * ( lambda_s *  tmp + np.dot(Rambda, m ))
    return m_hat

def main():
    random.seed(0)
    data = make_data(10)
    #plot_data(data)

    Rambda = np.identity(DIMMENSION)
    m = np.zeros(DIMMENSION)
    lambda_s = 1.0
    N =10

    r_hat = calc_r_hat(data[:N], lambda_s, Rambda)
    #r_hat_inv = np.transpose(r_hat)

    #m_hat = np.dot(r_hat_inv, np.dot(Rambda, m))
    m_hat = calc_m_hat(data[:N], r_hat, lambda_s, Rambda, m)

    plot_model(data[:N], r_hat, m_hat)

    print(r_hat, m_hat)

if __name__ == '__main__':
    main()


