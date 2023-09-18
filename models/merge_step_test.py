import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score,matthews_corrcoef

from pyod.models.ecod import ECOD
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.lunar import LUNAR


def prepare_input_train(file_path):
    file = pd.read_csv(file_path)
    input_array = file.iloc[:,7:]
    input_array.fillna(0,inplace = True)
    clade_list = file["nextstrain_clade"].values.tolist()
    return input_array,clade_list

def prepare_input_test(file_path,clade_list):
    file = pd.read_csv(file_path)
    input_array = file.iloc[:,7:]
    input_array.fillna(0,inplace = True)
    input_array = np.array(input_array)
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
    return input_array,target_list

def ecod_model(train_set,test_set,test_target):
    model = ECOD()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    matrix = confusion_matrix(test_target, pred_list, labels=[1, 0])
    if (matrix[1, 1] + matrix[1, 0]) == 0:
        specifity = 0
    else:
        specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    metrics_dict = {}
    metrics_dict["mcc"] = matthews_corrcoef(test_target, pred_list)
    metrics_dict["f1_score"] = f1_score(test_target, pred_list,pos_label=1)
    metrics_dict["precision"] = precision_score(test_target, pred_list,pos_label=1)
    metrics_dict["recall"] = recall_score(test_target, pred_list,pos_label=1)
    metrics_dict["accuracy"] = accuracy_score(test_target, pred_list)
    metrics_dict["specifity"] = specifity
    try:
        auc_value = roc_auc_score(test_target, pred_list)
        metrics_dict["auc"] = auc_value
    except ValueError:
        pass
    metrics_dict["condusion_matrix"] = matrix
    metrics_dict["result"] = pred_list
    return metrics_dict

def ocsvm_model(train_set,test_set,test_target):
    model = OCSVM()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    matrix = confusion_matrix(test_target, pred_list, labels=[1, 0])
    if (matrix[1, 1] + matrix[1, 0]) == 0:
        specifity = 0
    else:
        specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    metrics_dict = {}
    metrics_dict["mcc"] = matthews_corrcoef(test_target, pred_list)
    metrics_dict["f1_score"] = f1_score(test_target, pred_list,pos_label=1)
    metrics_dict["precision"] = precision_score(test_target, pred_list,pos_label=1)
    metrics_dict["recall"] = recall_score(test_target, pred_list,pos_label=1)
    metrics_dict["accuracy"] = accuracy_score(test_target, pred_list)
    metrics_dict["specifity"] = specifity
    try:
        auc_value = roc_auc_score(test_target, pred_list)
        metrics_dict["auc"] = auc_value
    except ValueError:
        pass
    metrics_dict["condusion_matrix"] = matrix
    metrics_dict["result"] = pred_list
    return metrics_dict

def knn_model(train_set,test_set,test_target):
    model = KNN()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    matrix = confusion_matrix(test_target, pred_list, labels=[1, 0])
    if (matrix[1, 1] + matrix[1, 0]) == 0:
        specifity = 0
    else:
        specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    metrics_dict = {}
    metrics_dict["mcc"] = matthews_corrcoef(test_target, pred_list)
    metrics_dict["f1_score"] = f1_score(test_target, pred_list,pos_label=1)
    metrics_dict["precision"] = precision_score(test_target, pred_list,pos_label=1)
    metrics_dict["recall"] = recall_score(test_target, pred_list,pos_label=1)
    metrics_dict["accuracy"] = accuracy_score(test_target, pred_list)
    metrics_dict["specifity"] = specifity
    try:
        auc_value = roc_auc_score(test_target, pred_list)
        metrics_dict["auc"] = auc_value
    except ValueError:
        pass
    metrics_dict["condusion_matrix"] = matrix
    metrics_dict["result"] = pred_list
    return metrics_dict

def iforest_model(train_set,test_set,test_target):
    model = IForest()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    matrix = confusion_matrix(test_target, pred_list, labels=[1, 0])
    if (matrix[1, 1] + matrix[1, 0]) == 0:
        specifity = 0
    else:
        specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    metrics_dict = {}
    metrics_dict["mcc"] = matthews_corrcoef(test_target, pred_list)
    metrics_dict["f1_score"] = f1_score(test_target, pred_list,pos_label=1)
    metrics_dict["precision"] = precision_score(test_target, pred_list,pos_label=1)
    metrics_dict["recall"] = recall_score(test_target, pred_list,pos_label=1)
    metrics_dict["accuracy"] = accuracy_score(test_target, pred_list)
    metrics_dict["specifity"] = specifity
    try:
        auc_value = roc_auc_score(test_target, pred_list)
        metrics_dict["auc"] = auc_value
    except ValueError:
        pass
    metrics_dict["condusion_matrix"] = matrix
    metrics_dict["result"] = pred_list
    return metrics_dict

def ae_model(train_set,test_set,test_target):
    model = AutoEncoder()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    matrix = confusion_matrix(test_target, pred_list, labels=[1, 0])
    if (matrix[1, 1] + matrix[1, 0]) == 0:
        specifity = 0
    else:
        specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    metrics_dict = {}
    metrics_dict["mcc"] = matthews_corrcoef(test_target, pred_list)
    metrics_dict["f1_score"] = f1_score(test_target, pred_list,pos_label=1)
    metrics_dict["precision"] = precision_score(test_target, pred_list,pos_label=1)
    metrics_dict["recall"] = recall_score(test_target, pred_list,pos_label=1)
    metrics_dict["accuracy"] = accuracy_score(test_target, pred_list)
    metrics_dict["specifity"] = specifity
    try:
        auc_value = roc_auc_score(test_target, pred_list)
        metrics_dict["auc"] = auc_value
    except ValueError:
        pass
    metrics_dict["condusion_matrix"] = matrix
    metrics_dict["result"] = pred_list
    return metrics_dict


def lunar_model(train_set,test_set,test_target):
    model = LUNAR()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    matrix = confusion_matrix(test_target, pred_list, labels=[1, 0])
    if (matrix[1, 1] + matrix[1, 0]) == 0:
        specifity = 0
    else:
        specifity = (matrix[1, 1]) / (matrix[1, 1] + matrix[1, 0])
    metrics_dict = {}
    metrics_dict["mcc"] = matthews_corrcoef(test_target, pred_list)
    metrics_dict["f1_score"] = f1_score(test_target, pred_list,pos_label=1)
    metrics_dict["precision"] = precision_score(test_target, pred_list,pos_label=1)
    metrics_dict["recall"] = recall_score(test_target, pred_list,pos_label=1)
    metrics_dict["accuracy"] = accuracy_score(test_target, pred_list)
    metrics_dict["specifity"] = specifity
    try:
        auc_value = roc_auc_score(test_target, pred_list)
        metrics_dict["auc"] = auc_value
    except ValueError:
        pass
    metrics_dict["condusion_matrix"] = matrix
    metrics_dict["result"] = pred_list
    return metrics_dict

if __name__ == "__main__":
    file_path = "D:/work files/stacking_model/divide_dataset/step1_dataset/"
    ecod_l = []
    ocsvm_l = []
    knn_l = []
    iforest_l = []
    ae_l = []
    lunar_l = []
    for i in range(1,4):
        train_path = file_path+"time_set_"+str(i)+"_train.csv"
        test_path = file_path+"time_set_"+str(i)+"_test.csv"
        train_set,clade_list= prepare_input_train(train_path)
        test_set,test_target = prepare_input_test(test_path,clade_list)
        print(i, "+1")
        metric_ecod = ecod_model(train_set,test_set,test_target)
        print(i, "+2")
        metric_ocsvm = ocsvm_model(train_set,test_set,test_target)
        print(i, "+3")
        metric_knn = knn_model(train_set,test_set,test_target)
        print(i, "+4")
        metric_iforest = iforest_model(train_set,test_set,test_target)
        print(i, "+5")
        metric_ae = ae_model(train_set,test_set,test_target)
        print(i, "+6")
        metric_lunar = lunar_model(train_set,test_set,test_target)
        ecod_l.append(metric_ecod)
        ocsvm_l.append(metric_ocsvm)
        knn_l.append(metric_knn)
        iforest_l.append(metric_iforest)
        ae_l.append(metric_ae)
        lunar_l.append(metric_lunar)
    ecod_f = pd.DataFrame(ecod_l)
    ocsvm_f = pd.DataFrame(ocsvm_l)
    knn_f = pd.DataFrame(knn_l)
    iforest_f = pd.DataFrame(iforest_l)
    ae_f = pd.DataFrame(ae_l)
    lunar_f = pd.DataFrame(lunar_l)
    ecod_f.to_csv("./merge_step_test/ecod.csv",index=False)
    ocsvm_f.to_csv("./merge_step_test/ocsvm.csv", index=False)
    knn_f.to_csv("./merge_step_test/knn.csv", index=False)
    iforest_f.to_csv("./merge_step_test/iforest.csv", index=False)
    ae_f.to_csv("./merge_step_test/ae.csv", index=False)
    lunar_f.to_csv("./merge_step_test/lunar.csv", index=False)
