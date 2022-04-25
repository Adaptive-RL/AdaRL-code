import numpy as np


def min_n_suff_set_state(sr, ss, thred=0.1):
    sr = np.absolute(sr)
    ss = np.absolute(ss)
    # s->r
    reduction_set1 = np.asarray(np.argwhere(sr > thred), dtype=int).reshape(-1)
    # s->s
    reduction_set2 = []
    for i in reduction_set1:
        for j in range(len(ss)):
            if ss[j, i] > thred:
                reduction_set2.append(j)
    reduction_set_s = np.unique(np.concatenate((reduction_set1, reduction_set2))).reshape(1, -1)
    return reduction_set_s


def min_n_suff_set_theta(reduction_set_s, ths, thred=0.1):
    if len(ths) == 1:
        return None
    else:
        reduction_set_th = []
        for s in reduction_set_s:
            for i in range(len(ths)):
                if ths[i][s] > thred:
                    reduction_set_th.append(i)
        return reduction_set_th
