import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file = pd.read_excel("./FPR_.xlsx")
KNN = file["KNN"]
LUNAR = file["LUNAR"]
KNN_LUNAR = file["KNN+LUNAR"]
KNN_KNN = file["KNN+KNN"]
LUNAR_LUNAR = file["LUNAR+LUNAR"]

def average(data):
    return sum(data) / len(data)

def bootstrap(data, b, c, func):
    array = np.array(data)
    n = len(array)
    sample_result_arr = []
    for i in range(b):
        index_arr = np.random.randint(0, n, size=n)
        data_sample = array[index_arr]
        sample_result = func(data_sample)
        sample_result_arr.append(sample_result)
    a = 1 - c
    k_1 = int(b * a / 2)
    k_2 = int(b * (1 - a / 2))
    AUC_sample_arr_sorted = sorted(sample_result_arr)
    lower = AUC_sample_arr_sorted[k_1]
    higher = AUC_sample_arr_sorted[k_2]

    return lower, higher


if __name__ == '__main__':
    result_KNN = bootstrap(KNN.tolist(), 1000, 0.95, average)
    result_LUNAR = bootstrap(LUNAR.tolist(), 1000, 0.95, average)
    result_KNN_LUNAR = bootstrap(KNN_LUNAR.tolist(), 1000, 0.95, average)
    result_KNN_KNN = bootstrap(KNN_KNN.tolist(), 1000, 0.95, average)
    result_LUNAR_LUNAR = bootstrap(LUNAR_LUNAR.tolist(), 1000, 0.95, average)
    print("KNN:",result_KNN)
    print("LUNAR:",result_LUNAR)
    print("KNN+LUNAR:",result_KNN_LUNAR)
    print("KNN+KNN:",result_KNN_KNN)
    print("LUNAR+LUNAR:",result_LUNAR_LUNAR)
