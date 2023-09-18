import pandas as pd

file = pd.read_csv("./counry_all_kmers_info_clear.csv")
neutral_set = file.loc[(file["classify_target"]=="neutral")]
positive_set = file.loc[(file["classify_target"]=="positive")]

m=0
for i in [1,10,100,1000,3000]:
    m = m+1
    train_set = neutral_set.sample(n=3000,random_state=i,replace=False)
    diff=pd.concat([neutral_set,train_set,train_set]).drop_duplicates(keep=False)
    test_neutral_set = diff.sample(n=150,random_state=i,replace=False)
    test_positive_set = positive_set.sample(n=150,random_state=i,replace=False)
    test_set = pd.concat([test_neutral_set,test_positive_set])
    path1 = "./step2_dataset/train_set_"+str(m)+".csv"
    path2 = "./step2_dataset/test_set_"+str(m)+".csv"
    train_set.to_csv(path1,index=False)
    test_set.to_csv(path2,index=False)
print(file)