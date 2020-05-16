import numpy as np
from matplotlib import pyplot as plt

# 真のポアソン分布のパラメータ
POI_λ1 = 30
POI_λ2 = 55
POI_SAMPLES_X1 = 200
POI_SAMPLES_X2 = 300

# サンプリング用のリスト
sample_s = []
sample_λ = []
sample_π = []

MAX_ITER = 100
K = 2

np.random.seed(123)

# パラメータ初期値
init_Gam_param_a = 1
init_Gam_param_b = 1
init_Dir_alpha = np.ones(K)





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

# 正規化に使う
# https://qiita.com/BigSea/items/1949b3ceefcec4fc32ea
def log_sum_exp(X):
    max_x = np.max(X, axis=1).reshape(-1, 1)
    return np.log(np.sum(np.exp(X - max_x), axis=1).reshape(-1, 1)) + max_x


def normalize_pi(k):
    # πの条件に従って正規化(各値を合計値で割る)
    Pi = np.ones(k)
    if np.sum(Pi) != 1:
        Pi = Pi / np.sum(Pi)

    return Pi

def main():
    # λ, πの初期値設定
    λ = np.ones(K)
    pi = normalize_pi(K)

    X = create_data(POI_λ1, POI_SAMPLES_X1, POI_λ2, POI_SAMPLES_X2)
    plot_data(X)
    N = len(X)

    for i in range(MAX_ITER):
        s = np.zeros((N, K))

        # 4.38
        log_eta = np.dot(X.reshape(N, 1), np.log(λ.reshape(1, K))) - λ.reshape(1, K) + np.log(pi.reshape(1, K))
        logsum_eta = -1 * log_sum_exp(log_eta)
        eta = np.exp(log_eta + logsum_eta)

        # ηをつかって、sをサンプリング 4.37
        for n in range(N):
            s[n] = np.random.multinomial(1, eta[n])
        sample_s.append(np.copy(s))

        # λのサンプリング 4.41
        gamma_a_hat = (np.dot(s.T, X.reshape(N, 1)) + init_Gam_param_a).T[0]
        gamma_b_hat = np.sum(s, axis=0).T + init_Gam_param_b

        # λのサンプル 4.41
        λ = np.random.gamma(gamma_a_hat, 1 / gamma_b_hat)
        sample_λ.append(np.copy(λ))

        # πのサンプルのためにαを計算
        α_hat = np.sum(s, axis=0) + init_Dir_alpha

        # αをパラメータとしてπをサンプル 4.41

        pi = np.random.dirichlet(α_hat)
        sample_π.append(np.copy(pi))

    sample_s_arr = np.array(sample_s)
    sample_lambda_arr = np.array(sample_λ)
    sample_pi_arr = np.array(sample_π)

    # 各クラスタの平均値
    ave_Lambda = list(np.average(sample_lambda_arr, axis=0))
    print(f'prediction: {ave_Lambda}')

    # 全データにおけるクラスタサンプル数の割合
    ave_Pi = list(np.average(sample_pi_arr, axis=0))

    all_samples = POI_SAMPLES_X1 + POI_SAMPLES_X2

    print(f'prediction: {ave_Pi}')


if __name__ == '__main__':
    main()
