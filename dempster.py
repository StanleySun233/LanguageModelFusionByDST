import numpy as np


def getK(a, b):
    return np.dot(a, b)


def dempster(a, b):
    k = getK(a, b)
    result = []
    for i in range(len(a)):
        _ = 1 / k * a[i] * b[i]
        result.append(_)
    return np.array(result)


if __name__ == '__main__':
    x = np.array([0.86, 0.13, 0.01])
    y = np.array([0.02, 0.90, 0.08])
    print(dempster(x, y))
