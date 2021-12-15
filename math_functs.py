import numpy as np

def high_mean(arr):
    uniques = np.unique(arr)
    highest_val = uniques[-1]

    total = 0
    counter = 1
    len_counter = 0
    for x in uniques:
        cnt = np.count_nonzero(arr == x)
        total += x*cnt*counter
        len_counter += cnt*counter
        counter += 1

    return total / len_counter