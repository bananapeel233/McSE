import numpy as np

def getTransferMatrix(A):
    '''
    :param A: initial matrix
    :return: transfer matrix
    '''
    tM = []
    for i in range(len(A)):
        if np.sum(A[i]) != 0:
            tM.append(A[i]/np.sum(A[i]))
        else:
            tM.append(A[i])
    tM = np.asarray(tM)
    return np.transpose(tM)


def pageRank(A, p=1, epochs=100, initial_type='avg'):
    '''
    :param A: initial matrix
    :param p: weight factor
    :return: PR
    '''
    if initial_type == 'all_one':
        initial = np.ones(len(A))
    elif initial_type == 'weighted':
        initial = []
        for i in range(len(A)):
            initial.append(np.sum(A[i]))
        initial = np.asarray(initial)
        print("initial before:", initial)
        if np.sum(initial) > 0:
            initial = initial / np.sum(initial)
        print("initial after:", initial)
    else:
        initial = np.ones(len(A))
        initial /= np.sum(initial)
    PR = initial.copy()
    # PR /= np.sum(PR)
    e = PR.copy()
    trans_m = getTransferMatrix(A)
    for i in range(epochs):
        PR = np.dot(trans_m, PR) * p + (1-p) * e

    return PR


def personalizedPageRank(A, p=0.85, epochs=100, initial_type='avg', specific_v=[]):
    '''
    :param A: initial matrix
    :param p: weight factor
    :return: PR
    '''
    if initial_type == 'all_one':
        initial = np.ones(len(A))
    elif initial_type == 'weighted':
        initial = []
        for i in range(len(A)):
            initial.append(np.sum(A[i]))
        initial = np.asarray(initial)
        if np.sum(initial) > 0:
            initial = initial / np.sum(initial)
    else:
        initial = np.ones(len(A))
        initial /= np.sum(initial)
    PR = initial.copy()
    trans_m = getTransferMatrix(A)
    for i in range(epochs):
        PR = np.dot(trans_m, PR) * p + (1-p) * specific_v
        

    return PR