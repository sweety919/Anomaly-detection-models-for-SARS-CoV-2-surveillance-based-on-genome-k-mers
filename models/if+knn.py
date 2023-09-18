from pyod.models.ecod import ECOD
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.lunar import LUNAR
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score,matthews_corrcoef


def ecod_model(train_set,test_set,test_target):
    model = ECOD()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    # matrix = confusion_matrix(test_target, pred_list, labels=[1, 0])
    # if (matrix[1, 1] + matrix[1, 0]) == 0:
    #     specifity = 0
    # else:
    #     specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    # metrics_dict = {}
    # metrics_dict["mcc"] = matthews_corrcoef(test_target, pred_list)
    # metrics_dict["f1_score"] = f1_score(test_target, pred_list,pos_label=1)
    # metrics_dict["precision"] = precision_score(test_target, pred_list,pos_label=1)
    # metrics_dict["recall"] = recall_score(test_target, pred_list,pos_label=1)
    # metrics_dict["accuracy"] = accuracy_score(test_target, pred_list)
    # metrics_dict["specifity"] = specifity
    # try:
    #     auc_value = roc_auc_score(test_target, pred_list)
    #     metrics_dict["auc"] = auc_value
    # except ValueError:
    #     pass
    # metrics_dict["condusion_matrix"] = matrix
    # metrics_dict["result"] = pred_list
    return pred_list

def ocsvm_model(train_set,test_set,test_target):
    model = OCSVM()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    # matrix = confusion_matrix(test_target, pred_list, labels=[1, 0])
    # if (matrix[1, 1] + matrix[1, 0]) == 0:
    #     specifity = 0
    # else:
    #     specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    # metrics_dict = {}
    # metrics_dict["mcc"] = matthews_corrcoef(test_target, pred_list)
    # metrics_dict["f1_score"] = f1_score(test_target, pred_list,pos_label=1)
    # metrics_dict["precision"] = precision_score(test_target, pred_list,pos_label=1)
    # metrics_dict["recall"] = recall_score(test_target, pred_list,pos_label=1)
    # metrics_dict["accuracy"] = accuracy_score(test_target, pred_list)
    # metrics_dict["specifity"] = specifity
    # try:
    #     auc_value = roc_auc_score(test_target, pred_list)
    #     metrics_dict["auc"] = auc_value
    # except ValueError:
    #     pass
    # metrics_dict["condusion_matrix"] = matrix
    # metrics_dict["result"] = pred_list
    return pred_list

def knn_model(train_set,test_set,test_target):
    model = KNN()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    # matrix = confusion_matrix(test_target, pred_list, labels=[1, 0])
    # if (matrix[1, 1] + matrix[1, 0]) == 0:
    #     specifity = 0
    # else:
    #     specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    # metrics_dict = {}
    # metrics_dict["mcc"] = matthews_corrcoef(test_target, pred_list)
    # metrics_dict["f1_score"] = f1_score(test_target, pred_list,pos_label=1)
    # metrics_dict["precision"] = precision_score(test_target, pred_list,pos_label=1)
    # metrics_dict["recall"] = recall_score(test_target, pred_list,pos_label=1)
    # metrics_dict["accuracy"] = accuracy_score(test_target, pred_list)
    # metrics_dict["specifity"] = specifity
    # try:
    #     auc_value = roc_auc_score(test_target, pred_list)
    #     metrics_dict["auc"] = auc_value
    # except ValueError:
    #     pass
    # metrics_dict["condusion_matrix"] = matrix
    # metrics_dict["result"] = pred_list
    return pred_list

def iforest_model(train_set,test_set,test_target):
    model = IForest()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    # matrix = confusion_matrix(test_target, pred_list, labels=[1, 0])
    # if (matrix[1, 1] + matrix[1, 0]) == 0:
    #     specifity = 0
    # else:
    #     specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    # metrics_dict = {}
    # metrics_dict["mcc"] = matthews_corrcoef(test_target, pred_list)
    # metrics_dict["f1_score"] = f1_score(test_target, pred_list,pos_label=1)
    # metrics_dict["precision"] = precision_score(test_target, pred_list,pos_label=1)
    # metrics_dict["recall"] = recall_score(test_target, pred_list,pos_label=1)
    # metrics_dict["accuracy"] = accuracy_score(test_target, pred_list)
    # metrics_dict["specifity"] = specifity
    # try:
    #     auc_value = roc_auc_score(test_target, pred_list)
    #     metrics_dict["auc"] = auc_value
    # except ValueError:
    #     pass
    # metrics_dict["condusion_matrix"] = matrix
    # metrics_dict["result"] = pred_list
    return pred_list

def ae_model(train_set,test_set,test_target):
    model = AutoEncoder()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    # matrix = confusion_matrix(test_target, pred_list, labels=[1, 0])
    # if (matrix[1, 1] + matrix[1, 0]) == 0:
    #     specifity = 0
    # else:
    #     specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    # metrics_dict = {}
    # metrics_dict["mcc"] = matthews_corrcoef(test_target, pred_list)
    # metrics_dict["f1_score"] = f1_score(test_target, pred_list,pos_label=1)
    # metrics_dict["precision"] = precision_score(test_target, pred_list,pos_label=1)
    # metrics_dict["recall"] = recall_score(test_target, pred_list,pos_label=1)
    # metrics_dict["accuracy"] = accuracy_score(test_target, pred_list)
    # metrics_dict["specifity"] = specifity
    # try:
    #     auc_value = roc_auc_score(test_target, pred_list)
    #     metrics_dict["auc"] = auc_value
    # except ValueError:
    #     pass
    # metrics_dict["condusion_matrix"] = matrix
    # metrics_dict["result"] = pred_list
    return pred_list


def lunar_model(train_set,test_set,test_target):
    model = LUNAR()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    # matrix = confusion_matrix(test_target, pred_list, labels=[1, 0])
    # if (matrix[1, 1] + matrix[1, 0]) == 0:
    #     specifity = 0
    # else:
    #     specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    # metrics_dict = {}
    # metrics_dict["mcc"] = matthews_corrcoef(test_target, pred_list)
    # metrics_dict["f1_score"] = f1_score(test_target, pred_list,pos_label=1)
    # metrics_dict["precision"] = precision_score(test_target, pred_list,pos_label=1)
    # metrics_dict["recall"] = recall_score(test_target, pred_list,pos_label=1)
    # metrics_dict["accuracy"] = accuracy_score(test_target, pred_list)
    # metrics_dict["specifity"] = specifity
    # try:
    #     auc_value = roc_auc_score(test_target, pred_list)
    #     metrics_dict["auc"] = auc_value
    # except ValueError:
    #     pass
    # metrics_dict["condusion_matrix"] = matrix
    # metrics_dict["result"] = pred_list
    return pred_list


def prepare_input_train_1(file_path):
    file = pd.read_csv(file_path)
    input_array = file.iloc[:,7:]
    input_array.fillna(0,inplace = True)
    input_array = np.array(input_array)
    clade_list = file["nextstrain_clade"].values.tolist()
    return input_array,clade_list

def prepare_input_train_2(file_path):
    file = pd.read_csv(file_path)
    neutral_list = []
    for i in range(len(file)):
        a = file.loc[i,"classify_target"]
        if a == "neutral":
            neutral_list.append((file.iloc[i]).tolist())
        else:
            continue
    neutral_list_frame = pd.DataFrame(neutral_list)
    input_array = neutral_list_frame.iloc[:,7:]
    input_array.fillna(0,inplace = True)
    input_array = np.array(input_array)
    return input_array

def prepare_input_test_1(file_path,clade_list):
    file = pd.read_csv(file_path)
    input_array = file.iloc[:,7:]
    input_array.fillna(0,inplace = True)
    input_array = np.array(input_array)
    target_list_ = file["nextstrain_clade"].values.tolist()
    target_list = []
    for i in target_list_:
        target = str(i).replace("[","").replace("]","").replace("'","")
        if target in clade_list:
            a=0
        else:
            a=1
        target_list.append(a)
    return input_array,target_list

def prepare_input_test_2(frame):
    input_array = frame.iloc[:, 7:]
    input_array.fillna(0, inplace=True)
    input_array = np.array(input_array)
    target_list_ = frame["classify_target"].values.tolist()
    target_list = []
    for i in target_list_:
        target = str(i).replace("[", "").replace("]", "").replace("'", "")
        if target == "neutral":
            a = 0
        else:
            a = 1
        target_list.append(a)
    return input_array, target_list

def deal_pred_result(result_1,result_2):
    comb_result = []
    k = 0
    for i in range(len(result_1)):
        if result_1[i]==0:
            comb_result.append(0)
        else:
            if result_2[k]==0:
                comb_result.append(0)
                k += 1
            else:
                comb_result.append(1)
                k += 1
    return comb_result

def deal_target(file_path,clade_list):
    file = pd.read_csv(file_path)
    target_list_ = file["nextstrain_clade"].values.tolist()
    target_list = []
    classify_list = file["classify_target"].values.tolist()
    for i in range(len(target_list_)):
        target = str(target_list_[i]).replace("[","").replace("]","").replace("'","")
        classify = str(classify_list[i]).replace("[","").replace("]","").replace("'","")
        if target in clade_list:
            a=0
        else:
            if classify=="positive":
                a=1
            else:
                a=0
        target_list.append(a)
    return target_list

def get_metrics(comb_result,comb_target):
    final_metrics = {}
    final_metrics["mcc"]=matthews_corrcoef(comb_target,comb_result)
    final_metrics["f1 score"]=f1_score(comb_target,comb_result,pos_label=1)
    final_metrics["precision"]=precision_score(comb_target,comb_result,pos_label=1)
    final_metrics["recall"]=recall_score(comb_target,comb_result,pos_label=1,)
    final_metrics["accuracy"]=accuracy_score(comb_target,comb_result)
    matrix = confusion_matrix(comb_target,comb_result, labels=[1, 0])
    if (matrix[1, 1] + matrix[1, 0]) == 0:
        specifity = 0
    else:
        specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    final_metrics["specifity"]=specifity
    try:
        auc_value = roc_auc_score(comb_target,comb_result)
        final_metrics["auc"] = auc_value
    except ValueError:
        pass
    final_metrics["target"]=comb_target
    final_metrics["result"]=comb_result
    return final_metrics

if __name__=="__main__":
    file_path = "D:/work files/stacking_model/divide_dataset/step1_dataset/"
    whole_metrics = []
    for i in range(1,4):
        train_path = file_path+"time_set_"+str(i)+"_train.csv"
        test_path = file_path+"time_set_"+str(i)+"_test.csv"
        train_set_1,clade_list= prepare_input_train_1(train_path)
        test_set_1,test_target_1 = prepare_input_test_1(test_path,clade_list)
        print(test_target_1)
        print(i, "+1")
        pred_list_1 = iforest_model(train_set_1, test_set_1, test_target_1)
        next_sample_index = [i for i,x in enumerate(pred_list_1) if x==1]
        pred_variant = []
        file = pd.read_csv(test_path)
        columns_name = file.columns
        for m in next_sample_index:
            row = (file.iloc[m]).tolist()
            pred_variant.append(row)
        pred_variant_DF = pd.DataFrame(pred_variant,columns=columns_name) #第二轮的测试集（初始）
        test_set_2, test_target_2 = prepare_input_test_2(pred_variant_DF)
        train_set_2 = prepare_input_train_2(train_path)  # 模型筛选时用的是制定的当年前期的数据集中的中性样本，因为数据量也相对够用
        print(i, "+2")
        pred_list_2 = knn_model(train_set_2, test_set_2, test_target_2)
        comb_target = deal_target(test_path,clade_list)
        comb_result = deal_pred_result(pred_list_1,pred_list_2)
        final_metrics = get_metrics(comb_result, comb_target)
        whole_metrics.append(final_metrics)
    whole_metrics_f = pd.DataFrame(whole_metrics)
    whole_metrics_f.to_csv("./stacking_test/iforest+knn.csv",index=False)
