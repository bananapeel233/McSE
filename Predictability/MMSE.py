from Predictability import *
from utils.load_data import *
import numpy as np
import time
import matplotlib.pyplot as plt
from kdtree import *
from tqdm import tqdm
import gc

def regionalize_data(timesery, percent=100, type='mid'):
    t_min = np.min(timesery)
    t_max = np.max(timesery)
    gap = (t_max - t_min) / percent
    if gap == 0:
        return timesery
    if type == 'left':
        timesery = ((timesery - t_min) // gap) * gap + t_min
    elif type == 'mid':
        timesery = ((timesery - t_min) // gap + 0.5) * gap + t_min
    return timesery


def normalize_data(raw_data):
    '''
    Normalize data to [-1, 1]
    :param raw_data: data before processing
    :return: Processed data
    '''

    data = np.zeros(raw_data.shape)
    for i in range(len(raw_data[0])):
        column_data = raw_data[:, i]
        scale = np.max(np.abs(column_data))
        if scale == 0:
            data[:, i] = column_data
        else:
            data[:, i] = column_data / scale
    return data


def getMaxVarIndex(data):
    # 获得方差最大维度的索引
    sort_index = np.argsort(np.var(data, axis=0))
    return sort_index[-1]


def MSampEn(L, m, tao, r, type='avg'):
    N = len(L[0])
    P = len(L)

    def _maxdist(x_i, x_j):
        return np.max(np.abs(x_i - x_j))

    def _phi_m(m):
        x = np.asarray([L[:, range(i, i + m * tao, tao)].reshape(-1) for i in range(N - (m - 1) * tao)])
        if len(x) <= 1:
            return 0

        base_index = getMaxVarIndex(x)

        tree = KDTree(len(x[0]))
        tree.root = tree.create(x.copy(), depth=0, base=base_index)
        B = 0
        for i in tqdm(range(len(x))):
            ranges = np.array([[v - r, v + r] for v in x[i]])
            B += tree.range_search(ranges)
        B -= len(x)
        B /= len(x) - 1
        del tree
        return B / len(x)

    B_m = _phi_m(m)
    B_m1 = _phi_m(m+1)
    if B_m == 0:
        return 10, -1
    gc.collect()
    return -np.log(B_m1 / B_m), 1


def MMSE(timeseiries, tao=1, m=2, r=0.25, type='avg'):
    var = np.var(timeseiries)
    return MSampEn(timeseiries, m, tao, r * var, type)


def Coarse(nlist, scale=3, p=100):
    if scale <= 1:
        return nlist
    Coarse_list = np.array([np.mean(nlist[:, i:i + scale], axis=1) for i in range(0, len(nlist[0]), scale)]).T
    return Coarse_list


a = np.array([[0, 0.5, 1, 1.5], [1, 1.5, 2, 2.5]])
print(Coarse(a, scale=2))


def Coarse_MMSE_predictability(dataname, nor_type='mid', nor_per=100, folder='', m=2,
                               tao=1, r=0.15, scale=3, max_N=1000000, max_T=1000000, MMSE_type='all', speed='kdtree'):
    dataset = '../dataset/' + str(dataname)
    data = load_data(dataset, type=dataset[-3:])
    data = np.array(data)
    start = time.time()

    timeseries = data.T
    if timeseries.shape[0] > max_N:
        timeseries = timeseries[:max_N, :]
    if timeseries.shape[1] > max_T:
        timeseries = timeseries[:, :max_T]

    data = Coarse(timeseries, scale, nor_per)
    timeseries = normalize_data(data.T).T

    entropy, tag = MMSE(timeseries, tao, m, r, MMSE_type)
    pred = maximum_predictability(len(np.unique(timeseries)), entropy)
    if pred == 'No solutions':
        pred = 0

    end = time.time()
    print("entropy:\t", entropy)
    print("preds:\t", pred)
    print("time:\t", end - start)

    return entropy, pred

