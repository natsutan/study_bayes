import numpy as np
from matplotlib import pyplot as plt

# 真のポアソン分布のパラメータ
POI_λ1 = 30
POI_λ2 = 50
POI_SAMPLES_X1 = 200
POI_SAMPLES_X2 = 300

np.random.seed(123)


def create_data(lambda1, n1, lambda2, n2):
    x1 = np.random.poisson(lambda1, n1)
    x2 = np.random.poisson(lambda2, n2)

    # データの結合
    X = np.concatenate([x1, x2])
    return X


def main():

    X = create_data(POI_λ1, POI_SAMPLES_X1, POI_λ2, POI_SAMPLES_X2)

    plt.figure(figsize=(12, 5))
    plt.title('Poisson distribution', fontsize=20)

    plt.hist(X, 30, density=True, label='all data')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
