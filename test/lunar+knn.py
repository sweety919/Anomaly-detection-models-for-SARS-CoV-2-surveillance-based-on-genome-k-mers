import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from pyod.models.knn import KNN
from pyod.models.lunar import LUNAR
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score,matthews_corrcoef



def prepare_file(country,date):
    that_day = file.loc[(file["country"]==country)&(file["collection_time"]==date)]
    date_ = datetime.strptime(date, '%Y-%m-%d')
    date_30_ = date_+timedelta(days=-30) #时间格式，变体出现前30日
    date_30 = date_30_.strftime('%Y-%m-%d')
    date_num = int(date.replace("-", ""))
    date_30_num = int(date_30.replace("-", ""))
    before_that_day = file.loc[(file["country"] == country) & (file["c_t_num"]<date_num)&(file["c_t_num"]>=date_30_num)]
    return that_day,before_that_day # 测试集、训练集

def prepare_file_2(country,date):
    date_ = datetime.strptime(date, '%Y-%m-%d')  # 时间格式，变体出现当日
    date_1_ = date_ + timedelta(days=-3)  # 时间格式，变体出现前3日
    date_1 = date_1_.strftime('%Y-%m-%d')
    date_2_ = date_ + timedelta(days=3)  # 时间格式，变体出现后3日
    date_2 = date_2_.strftime('%Y-%m-%d')
    date_30_ = date_ + timedelta(days=-33)  # 时间格式，训练集第一天
    date_30 = date_30_.strftime('%Y-%m-%d')
    date_1_num = int(date_1.replace("-", ""))
    date_2_num = int(date_2.replace("-", ""))
    date_30_num = int(date_30.replace("-", ""))
    test_days = file.loc[(file["country"] == country) & (file["c_t_num"] <= date_2_num) & (
                file["c_t_num"] >= date_1_num)]
    train_days = file.loc[(file["country"] == country) & (file["c_t_num"] < date_1_num) & (
                file["c_t_num"] >= date_30_num)]
    return test_days,train_days



def prepare_input_train_1(file):
    input_array = file.iloc[:,8:]
    input_array.fillna(0,inplace = True)
    input_array = np.array(input_array)
    clade_list = file["nextstrain_clade"].values.tolist()
    return input_array,clade_list

def prepare_input_train_2(country,date):
    pre_file = file.loc[(file["country"] == country) & (file["c_t_num"] <= int(date.replace("-", "")))]
    pre_file = pre_file.reset_index(drop=True)
    neutral_list = []
    for i in range(len(pre_file)):
        a = pre_file.loc[i,"classify_target"]
        if a == "neutral":
            neutral_list.append((file.iloc[i]).tolist())
        else:
            continue
    neutral_list_frame = pd.DataFrame(neutral_list)
    input_array = neutral_list_frame.iloc[:,8:]
    input_array.fillna(0,inplace = True)
    input_array = np.array(input_array)
    return input_array

def prepare_input_test_1(file,clade_list):
    input_array = file.iloc[:,8:]
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
    input_array = frame.iloc[:, 8:]
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
    return pred_list,metrics_dict

def knn_model_2(train_set,test_set):
    model = KNN()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    return pred_list


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
    return pred_list,metrics_dict

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

def deal_target(file,clade_list):
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
    final_metrics["pred_v_num"] = comb_result.count(1)
    return final_metrics


if __name__ == "__main__":
    file = pd.read_csv("./counry_all_kmers_info_clear_.csv")
    info_file = pd.read_csv("./pre_count_all(30-1).csv")
    whole_metrics = []
    knn_l = []
    for i in range(0,len(info_file)):
        row = info_file.loc[i]
        country = row["country"]
        clade = row["clade"]
        date = row["date"]
        test_file,train_file=prepare_file(country,date)
        # test_file, train_file = prepare_file_2(country, date)  # 测试集7天训练集30天
        train_set_1, clade_list = prepare_input_train_1(train_file)
        test_set_1, test_target_1 = prepare_input_test_1(test_file, clade_list)
        if len(train_set_1)==0:
            continue
        elif (len(train_set_1)<=5)&(len(train_set_1)<len(test_set_1)):  # 防止knn划分不了聚类
            continue
        else:
            print(i, "+1")
            pred_list_1, metric_knn = lunar_model(train_set_1, test_set_1, test_target_1)
            knn_l.append(metric_knn)
            # pred_list_1 = knn_model(train_set_1, test_set_1)
            if pred_list_1.tolist().count(1) == 0:
                comb_result = [0] * len(pred_list_1)
            else:
                next_sample_index = [i for i, x in enumerate(pred_list_1) if x == 1]
                pred_variant = []
                columns_name = test_file.columns
                for m in next_sample_index:
                    row = (test_file.iloc[m]).tolist()
                    pred_variant.append(row)
                pred_variant_DF = pd.DataFrame(pred_variant, columns=columns_name)  # 第二轮的测试集（初始）
                test_set_2, test_target_2 = prepare_input_test_2(pred_variant_DF)
                train_set_2 = prepare_input_train_2(country, date)  # 区别于模型筛选时用的是制定的当年前期的数据集中的中性样本，这里用的是这个时间点以前该国所有的中性样本
                print(i, "+2")
                pred_list_2 = knn_model_2(train_set_2, test_set_2)
                comb_result = deal_pred_result(pred_list_1, pred_list_2)
            comb_target = deal_target(test_file, clade_list)
            final_metrics = get_metrics(comb_result, comb_target)
            whole_metrics.append(final_metrics)

    whole_metrics_f = pd.DataFrame(whole_metrics)
    whole_metrics_f.to_csv("./test_lunar+knn(n_neighbors=5)0905.csv")
    knn_f = pd.DataFrame(knn_l)
    knn_f.to_csv("./step1_lunar(lunar+knn)0905.csv")
    print(file)
