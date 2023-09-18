import pandas as pd
import numpy as np
from datetime import datetime,timedelta

from pyod.models.knn import KNN
from pyod.models.lunar import LUNAR
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score,matthews_corrcoef


file = pd.read_csv("D:/work files/stacking_model/test/counry_all_kmers_info_clear_.csv")

def prepare_data(country,date): # 训练集30天测试集1天
    date_30 = date + timedelta(days=-30)
    date_num = int(date.strftime('%Y-%m-%d').replace("-",""))
    date_30_num = int(date_30.strftime('%Y-%m-%d').replace("-",""))
    test_sample = file.loc[(file["c_t_num"]==date_num)&(file["country"]==country)]
    test_sample_num = len(test_sample)
    train_sample = file.loc[(file["c_t_num"]<date_num)&(file["c_t_num"]>=date_30_num)&(file["country"]==country)]
    train_sample_num = len(train_sample)
    sample_count = {}
    sample_count["country"] = country
    sample_count["date"] = date.strftime('%Y-%m-%d')
    sample_count["test set"]=test_sample_num
    sample_count["train set"]=train_sample_num
    return sample_count,test_sample,train_sample


def prepare_input_train_1(data_file):
    input_array = data_file.iloc[:,8:]
    input_array.fillna(0,inplace = True)
    input_array = np.array(input_array)
    clade_list = data_file["nextstrain_clade"].values.tolist()
    return input_array,clade_list

def prepare_input_train_2(country,date):
    pre_file = file.loc[(file["country"] == country) & (file["c_t_num"] <= int(date.strftime('%Y-%m-%d').replace("-", "")))]
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


def prepare_input_test_1(data_file,clade_list):
    input_array = data_file.iloc[:,8:]
    input_array.fillna(0,inplace = True)
    input_array = np.array(input_array)
    target_list_ = data_file["nextstrain_clade"].values.tolist()
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
    model = KNN(n_neighbors=2)
    model.fit(train_set)
    pred_list = model.predict(test_set)
    return pred_list


def lunar_model(train_set,test_set):
    model = LUNAR()
    model.fit(train_set)
    pred_list = model.predict(test_set)
    return pred_list

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

def get_metrics(comb_result,comb_target,country,date):
    final_metrics = {}
    final_metrics["country"]=country
    final_metrics["date"]=date.strftime('%Y-%m-%d')
    final_metrics["pred_v_num"] = comb_result.count(1)
    final_metrics["real_v_num"] = comb_target.count(1)
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



if __name__ == "__main__":

    country_list = ["China","Argentina","Portugal"]

    for country in country_list:
        whole_metrics = []
        sample_list = []
        path1 = country+"_days.xlsx"
        day_file = pd.read_excel(path1)
        for i in range(len(day_file)):
            date = (day_file.loc[i])["date"]
            sample_count, test_sample, train_sample = prepare_data(country,date)
            sample_list.append(sample_count)
            train_set_1, clade_list = prepare_input_train_1(train_sample)
            test_set_1, test_target_1 = prepare_input_test_1(test_sample, clade_list)
            if len(train_set_1) == 0:
                continue
            elif len(test_set_1)==0:
                continue
            else:
                print(i, "+1")
                pred_list_1 = knn_model(train_set_1, test_set_1, test_target_1)
                # pred_list_1 = knn_model(train_set_1, test_set_1)
                if pred_list_1.tolist().count(1) == 0:
                    comb_result = [0] * len(pred_list_1)
                else:
                    next_sample_index = [i for i, x in enumerate(pred_list_1) if x == 1]
                    pred_variant = []
                    columns_name = test_sample.columns
                    for m in next_sample_index:
                        row = (test_sample.iloc[m]).tolist()
                        pred_variant.append(row)
                    pred_variant_DF = pd.DataFrame(pred_variant, columns=columns_name)  # 第二轮的测试集（初始）
                    test_set_2, test_target_2 = prepare_input_test_2(pred_variant_DF)
                    train_set_2 = prepare_input_train_2(country,date)  # 这里用的是这个时间点以前该国所有的中性样本
                    print(i, "+2")
                    pred_list_2 = lunar_model(train_set_2, test_set_2)
                    comb_result = deal_pred_result(pred_list_1, pred_list_2)
            comb_target = deal_target(test_sample, clade_list)
            final_metrics = get_metrics(comb_result, comb_target,country,date)
            whole_metrics.append(final_metrics)
        whole_metrics_f = pd.DataFrame(whole_metrics)
        path2_1 = country+"_0904_knn+lunar.csv"
        path2_2 = country+"_0904_num.csv"
        whole_metrics_f.to_csv(path2_1)
        sample_list_f = pd.DataFrame(sample_list)
        sample_list_f.to_csv(path2_2)
    print(file)