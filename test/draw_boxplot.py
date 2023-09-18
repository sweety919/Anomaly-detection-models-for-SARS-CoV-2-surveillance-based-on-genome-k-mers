import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

labels = 'MCC', 'F1_score', 'Precision', 'Recall', 'Accuracy', 'Specificity', 'AUC'
def draw_boxplot(file,name):
    file = file.fillna(value="empty")
    mcc_ = file["MCC"].values.tolist()
    mcc = [i for i in mcc_ if i != "empty"]
    f1score_ = file["F1_score"].values.tolist()
    f1score = [i for i in f1score_ if i != "empty"]
    precision_ = file["Precision"].values.tolist()
    precision = [i for i in precision_ if i != "empty"]
    recall_ = file["Recall"].values.tolist()
    recall = [i for i in recall_ if i != "empty"]
    accuracy_ = file["Accuracy"].values.tolist()
    accuracy = [i for i in accuracy_ if i != "empty"]
    specificity_ = file["Specificity"].values.tolist()
    specificity = [i for i in specificity_ if i != "empty"]
    auc_ = file["AUC"].values.tolist()
    auc = [i for i in auc_ if i != "empty"]
    plt.title(name, fontsize=10)
    plt.grid(True)
    plt.boxplot([mcc,f1score,precision,recall,accuracy,specificity,auc],
            medianprops={'color': 'red', 'linewidth': 1.5},
            meanline=True,
            showmeans=True,
            meanprops={'color': 'blue', 'ls': '--', 'linewidth': 1.5},
            flierprops={"marker": "o", "markerfacecolor": "#ff8884", "markersize": 3,"markeredgecolor":"#ff8884"},
            labels=labels)
    plt.yticks(np.arange(-0.3,1.2,0.1))
    path = name+".png"
    plt.savefig(path)
    plt.show()
    return

if __name__ == "__main__":
    # file_name = ["KNN","LUNAR","KNN+LUNAR","KNN+KNN","LUNAR+KNN"]
    file_name = ["step1KNN", "step1LUNAR"]
    for name in file_name:
        path = "./结果整理拆分/"+name+".xlsx"
        file = pd.read_excel(path)
        draw_boxplot(file,name)


