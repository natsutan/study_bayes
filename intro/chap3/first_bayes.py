import random
import math
import numpy as np

N_MAX = 10
TRUE_MU = 0.25

def a_hat(xs, a):
    return np.sum(xs) + a


def b_hat(xs, b):
    N = len(xs)
    return N - np.sum(xs) + b


def coin_toss():
    if random.random() < TRUE_MU:
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

def plot_beta_distribution(a, b):
    x = np.arange(0.01, 1.0, 0.01)
    y = beta(x, a, b)
    print(y)


def main():
    ns = []
    for i in range(N_MAX):
        coin = coin_toss()
        ns.append(coin)

    plot_beta_distribution(0.5, 0.5)



    print(ns)


if __name__ == '__main__':
    main()
