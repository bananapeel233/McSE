from Predictability import *
from utils.load_data import *
import numpy as np
import time
import matplotlib.pyplot as plt
import kdtree
from tqdm import tqdm
import gc
import pageRank as pr
import UnionSet as US



def regionalize_data(timesery, percent=100, type='mid'):
    t_min = np.min(timesery)
    t_max = np.max(timesery)
    gap = round((t_max - t_min) / percent, 2)
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


def softmax(f):
    f -= np.max(f)
    return np.exp(f) / np.sum(np.exp(f))


def McSampEn(L, m, tao, r, Matrix=[], normalized=1, pr_p=0.85, pr_initial_type='avg', topk=1, sv=[]):
    N = len(L[0])
    P = len(L)

    M = Matrix.copy()
    PRs = pr.personalizedPageRank(M, p=pr_p, initial_type=pr_initial_type, specific_v=sv)

    if len(np.unique(PRs)) == 1:
        PRs = np.ones(len(PRs))


    PR_index = np.argsort(PRs)[-topk:]
    L = L[PR_index, :]
    P = len(L)
    PRs = PRs[PR_index]
    PRs /= np.sum(PRs)
    Weights = np.ones(P)
    Weights += PRs

    def _phi_m(m):
        x = np.asarray([L[:, range(i, i + m * tao, tao)].reshape(-1) for i in range(N - (m - 1) * tao)])
        if len(x) <= 1:
            return 0

        W = np.asarray([Weights[i//m] for i in range(len(x[0]))])
        Bm = 0
        base_index = getMaxVarIndex(x)

        tree = kdtree.KDTree(len(x[0]))
        tree.root = tree.create(x.copy(), depth=0, base=base_index)

        # B = [(len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) - 1.0) / (N - m) for x_i in x]
        B = 0
        for i in range(len(x)):
            ranges = np.array([[x[i][j] - W[j] * r, x[i][j] + W[j] * r] for j in range(len(x[i]))])
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


def McSE(timeseiries, tao=1, m=2, r=0.25, Matrix=[], normalized=1, pr_p=0.85, pr_initial_type='avg', topk=1, sv=[]):
    var = np.var(timeseiries)
    return McSampEn(timeseiries, m, tao, r * var, Matrix, normalized, pr_p, pr_initial_type, topk, sv)


def Coarse(nlist, scale=3, p=100):
    if scale <= 1:
        return nlist
    Coarse_list = np.array([np.mean(nlist[:, i:i + scale], axis=1) for i in range(0, len(nlist[0]), scale)]).T
    return Coarse_list



def Coarse_McSE_predictability(dataname, nor_type='mid', nor_per=100, folder='', m=2, matrixName='', matrix_nor=1, pr_p = 0.85,
                               tao=1, r=0.15, scale=3, max_N=1000000, max_T=1000000, MMSE_type='all', speed='kdtree',
                               resultname='', al_type='kdtree', pr_initial_type='avg', topk=1):
    dataset = '../dataset/' + str(dataname)
    data = load_data(dataset, type=dataset[-3:])
    data = np.array(data)

    timeseries = data.T
    if timeseries.shape[0] > max_N:
        timeseries = timeseries[:max_N, :]
    if timeseries.shape[1] > max_T:
        timeseries = timeseries[:, :max_T]

    data = Coarse(timeseries, scale, nor_per)
    timeseries = normalize_data(data.T).T

    Matrix = np.asarray(load_data(matrixName, type='csv'))

    if len(Matrix) > max_N:
        Matrix = Matrix[:max_N, :max_N]


    if topk > max_N:
        topk = max_N
    entropies = []
    preds = []
    us = US.unionSet(Matrix)
    us.visualize()

    start = 0

    for i in tqdm(range(start, len(timeseries))):
        M_index = us.get_subConnectedGraph(i)
        print("subgraph for i:", M_index)
        if i not in M_index:
            M_index = np.append(M_index, i)
        M = []
        L = len(M_index)
        for j in range(L):
            for k in range(L):
                M.append(Matrix[M_index[j]][M_index[k]])
        M = np.asarray(M)
        M = M.reshape((L, L))
        t = timeseries[M_index, :]
        sv = np.zeros(len(M))
        i_index = np.where(M_index == i)[0][0]
        sv[i_index] = 1
        k = topk
        if k > L:
            k = L

        entropy, tag = McSE(t, tao, m, r, M, matrix_nor, pr_p, pr_initial_type, k, sv)
        pred = maximum_predictability(len(np.unique(t)), entropy)

        if pred == 'No solutions':
            pred = 0
        entropies.append(entropy)
        preds.append(pred)

    entropies = np.asarray(entropies)
    preds = np.asarray(preds)

    return np.mean(entropies), np.mean(preds)