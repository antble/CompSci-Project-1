import numpy as np

def bootstrap(x, y, sample_ratio):
    n = np.size(y)
    sample_size = int(np.floor(sample_ratio*n))
    train_ind = np.zeros(sample_size, dtype=int)
    ind = np.arange(0,n)
    train_ind = np.random.choice(ind, sample_size)
    test_ind = np.delete(ind, train_ind)

    x_train = x[train_ind]
    x_test = x[test_ind]
    y_train = y[train_ind]
    y_test = y[test_ind]

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x = np.linspace(0, 1, 10)
    y = np.linspace(3, 5, 10)
    sample_ratio = 0.31
    print(bootstrap(x,y,sample_ratio))

