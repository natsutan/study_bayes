import random
import math
import numpy as np

N_MAX = 10
TRUE_MU = 0.25

def a_hat(xs, a):
    return 0


def b_hat(xs, b):
    return 0


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


def plot_bernoulli(mu):
    x = np.arange(0.0, 1.0, 0.01)
    y = bernoulli(x, mu)
    print(y)




def main():
    ns = []
    for i in range(N_MAX):
        coin = coin_toss()
        ns.append(coin)

    plot_bernoulli(0.5)



    print(ns)


if __name__ == '__main__':
    main()
