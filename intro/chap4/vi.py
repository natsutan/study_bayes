import numpy as np
from scipy.stats import poisson
from scipy.special import psi
from matplotlib import pyplot as plt

# 真のポアソン分布のパラメータ
POI_λ1 = 33
POI_λ2 = 55
POI_SAMPLES_X1 = 200
POI_SAMPLES_X2 = 300

# サンプリング用のリスト
sample_s = []
sample_λ = []
sample_π = []

MAX_ITER = 100
K = 2

np.random.seed(1234)

def create_data(lambda1, n1, lambda2, n2):
    x1 = np.random.poisson(lambda1, n1)
    x2 = np.random.poisson(lambda2, n2)

    # データの結合
    X = np.concatenate([x1, x2])
    return X

def plot_data(x):
    plt.figure(figsize=(12, 5))
    plt.title('Poisson distribution', fontsize=20)

    plt.hist(x, 30, density=True, label='all data')
    plt.legend()
    plt.show()

def log_sum_exp(X):
    max_x = np.max(X, axis=1).reshape(-1, 1)
    return np.log(np.sum(np.exp(X - max_x), axis=1).reshape(-1, 1)) + max_x


def normalize_pi(k):
    # πの条件に従って正規化(各値を合計値で割る)
    Pi = np.ones(k)
    if np.sum(Pi) != 1:
        Pi = Pi / np.sum(Pi)


def main():
    # λ, πの初期値設定
    λ = np.ones(K)
    pi = normalize_pi(K)

    X = create_data(POI_λ1, POI_SAMPLES_X1, POI_λ2, POI_SAMPLES_X2)
    plot_data(X)
    N = len(X)

    init_a = np.ones(K)
    init_b = np.ones(K)
    init_alpha = np.ones(K)

    # 初期値いじる
    init_a[0] = 25
    init_a[1] = 60


    a_hat = np.copy(init_a)
    b_hat = np.copy(init_b)
    alpha_hat = np.copy(init_alpha)

    for i in range(MAX_ITER):
        # Estep
        # 4.6.0
        E_λ = a_hat / b_hat
        E_logλ = psi(a_hat) - np.log(b_hat)
        E_logπ = psi(alpha_hat) - psi(np.sum(alpha_hat))

        # 4.5.1
        log_eta = np.dot(X.reshape(N, 1), E_logλ.reshape(1, K)) - E_λ.reshape(1, K) + E_logπ.reshape(1, K)
        logsum_eta = -1 * log_sum_exp(log_eta)
        eta = np.exp(log_eta + logsum_eta)

        # etaからE_sを作りたい
        # E_s = eta 4.5.9
        E_s = eta

        a_hat = np.dot(X, E_s) + init_a
        b_hat = np.sum(E_s, axis=0) + init_b
        alpha_hat = np.sum(E_s, axis=0) + init_alpha
        print(a_hat)

    print(a_hat/b_hat, b_hat)




if __name__ == '__main__':
    main()
