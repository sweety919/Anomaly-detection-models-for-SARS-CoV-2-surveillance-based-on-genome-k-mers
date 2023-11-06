import pandas as pd
import matplotlib.pyplot as plt


file = pd.read_excel("./FPR_.xlsx")
KNN = file["KNN"]
LUNAR = file["LUNAR"]
KNN_LUNAR = file["KNN+LUNAR"]
KNN_KNN = file["KNN+KNN"]
LUNAR_LUNAR = file["LUNAR+LUNAR"]

# KNN01 = KNN.value_counts() # 计数
# plt.scatter(KNN01.index,KNN01.values)
# plt.show()
#
# LUNAR01 = LUNAR.value_counts() # 计数
# plt.scatter(LUNAR01.index,LUNAR01.values)
# plt.show()
#
# KNN_LUNAR01 = KNN_LUNAR.value_counts() # 计数
# plt.scatter(KNN_LUNAR01.index,KNN_LUNAR01.values)
# plt.show()
#
# KNN_KNN01 = KNN_KNN.value_counts() # 计数
# plt.scatter(KNN_KNN01.index,KNN_KNN01.values)
# plt.show()
#
# LUNAR_LUNAR01 = LUNAR_LUNAR.value_counts() # 计数
# plt.scatter(LUNAR_LUNAR01.index,LUNAR_LUNAR01.values)
# plt.show()

import numpy as np


def average(data):
    return sum(data) / len(data)


def bootstrap(data, B, c, func):
    """
    计算bootstrap置信区间
    :param data: array 保存样本数据
    :param B: 抽样次数 通常B>=1000
    :param c: 置信水平
    :param func: 样本估计量
    :return: bootstrap置信区间上下限
    """
    array = np.array(data)
    n = len(array)
    sample_result_arr = []
    for i in range(B):
        index_arr = np.random.randint(0, n, size=n)
        data_sample = array[index_arr]
        sample_result = func(data_sample)
        sample_result_arr.append(sample_result)

    a = 1 - c
    k1 = int(B * a / 2)
    k2 = int(B * (1 - a / 2))
    auc_sample_arr_sorted = sorted(sample_result_arr)
    lower = auc_sample_arr_sorted[k1]
    higher = auc_sample_arr_sorted[k2]

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