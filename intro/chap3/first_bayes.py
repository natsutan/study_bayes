import random
import math
import numpy as np
import matplotlib.pyplot as plt

N_MAX = 100
TRUE_MU = 0.3


def a_hat(xs, a):
    return np.sum(xs) + a


def b_hat(xs, b):
    N = len(xs)
    return N - np.sum(xs) + b


def coin_toss():
    if random.random() > TRUE_MU:
        return 0
    else:
        return 1


def bernoulli(x, mu):
    a = np.power(mu, x)
    b = np.power((1-mu), (1-x))
    c = a * b.T
    return np.power(mu, x) * np.power((1-mu), (1-x))


def beta(x, a, b):
    c_b_inv = (math.gamma(a) * math.gamma(b)) / math.gamma(a + b)

    t1 = np.power(x, a-1)
    t2 = np.power(1-x, b-1)

    return (t1 * t2) / c_b_inv

def calc_beta_distribution(a, b):
    x = np.arange(0.01, 1.0, 0.01)
    y = beta(x, a, b)
    return x, y


def main():
    ns = []
    a = 0.5
    b = 0.5
    x, y = calc_beta_distribution(a, b)
    plt.plot(x, y)
    plt.savefig('img/fig%03d.png' % 0)
    plt.clf()

    for i in range(N_MAX):
        coin = coin_toss()
        ns.append(coin)

        a_ = a_hat(ns, a)
        b_ = b_hat(ns, b)

        if (i % 10) == 0:
            x, y = calc_beta_distribution(a_, b_)
            plt.plot(x,y)
            plt.savefig('img/fig%03d.png' % i)
            plt.clf()

    print(a_, b_)

    plt.show()

    print(ns)


if __name__ == '__main__':
    main()
