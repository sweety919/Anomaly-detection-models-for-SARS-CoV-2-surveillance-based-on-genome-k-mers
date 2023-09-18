from collections import Counter
from functools import reduce
import pandas as pd


def load_fa(path):
    """a function to read fasta file from the path and store in a dict"""
    genes_seq = {}  #将序列存入字典
    with open(path,"r") as sequences:  #以读取方式打开文件
        lines = sequences.readlines()

    for line in lines:
        if line.startswith(">"):
            # genename = line.split()[1]  #这个地方需要灵活调整
            genename = line.replace(">","").replace("\n","")
            genes_seq[genename] = ''  #序列为字符串
        else:
            genes_seq[genename] += line.strip()

    return genes_seq

def build_kmers(seq, k_size):
    """a function to calculate kmers from seq"""
    kmers = []  # k-mer存储在列表中
    n_kmers = len(seq) - k_size + 1

    for i in range(n_kmers):
        kmer = seq[i:i + k_size]
        kmers.append(kmer)

    return kmers

def summary_kmers(kmers):
    """a function to summarize the kmers"""
    kmers_stat = dict(Counter(kmers))
    return kmers_stat

def kmers_main(kmers_size,file_name): # 采用kmers vector的思路
    file_path = "D:/work files/gisaid_dataset/" + file_name + "/all_fasta.fasta"
    genes_seq = load_fa(file_path)
    genes_kmers = {}
    for gene in genes_seq.keys():
        genes_kmers[gene] = summary_kmers(build_kmers(seq=genes_seq[gene], k_size=kmers_size))
    kmers_frame_pre = pd.DataFrame(genes_kmers)
    nucleotide = ["A","T","C","G"]
    kmers_target = reduce(lambda x, y:[z0 + z1 for z0 in x for z1 in y], [nucleotide] * kmers_size)
    pre_index = list(kmers_frame_pre.index)
    kmers_frame_l = []
    for row in (kmers_frame_pre.T).itertuples():
        kmer_single_seq = []
        for i in range(len(kmers_target)):
            a = kmers_target[i]
            try:
                b = pre_index.index(a)
                kmer_num = row[(b + 1)]
            except ValueError:
                kmer_num = 0
            kmer_single_seq.append(kmer_num)
        kmers_frame_l.append(kmer_single_seq)
    kmers_frame = pd.DataFrame(kmers_frame_l)
    kmers_frame.columns = [kmers_target]
    kmers_frame.index = [kmers_frame_pre.columns]
    return kmers_frame

#下列函数使用于China
def frame_clear_1(kmers_frame,file_name):
    file_path = "D:/work files/gisaid_dataset/"+file_name+"/nextclade_simple.xlsx"
    nextclade_tsv = pd.read_excel(file_path)
    kmers_frame_reset_ = kmers_frame.reset_index()
    kmers_frame_reset = kmers_frame_reset_.rename(columns={"level_0": "name"})
    name_list = kmers_frame_reset["name"].values.tolist()
    accession_id_list = []
    collection_time_list = []
    nextstrain_clade_list = []
    for row in name_list:
        row = str(row).replace("[","").replace("]","").replace("'","")
        clade = str(nextclade_tsv.loc[(nextclade_tsv["seqName"]==row)]["clade"].tolist()).replace("'","").replace("[","").replace("]","")
        accession_id = row.split("|")[1]
        collection_time = row.split("|")[2]
        accession_id_list.append(accession_id)
        collection_time_list.append(collection_time)
        nextstrain_clade_list.append(clade)
    # kmers_frame_reset = kmers_frame_reset.rename(columns={"level_0":"name"})
    kmers_frame_reset.insert(1,"accession_id",accession_id_list)
    kmers_frame_reset.insert(2,"collection_time",collection_time_list)
    kmers_frame_reset.insert(3,"nextstrain_clade",nextstrain_clade_list)
    info_kmers_frame = kmers_frame_reset
    return info_kmers_frame

#下列函数适用于Portugal和Argentina
def frame_clear_2(kmers_frame,file_name):
    file_path_1 = "D:/work files/gisaid_dataset/"+file_name+"/nextclade_simple.xlsx"
    nextclade_tsv = pd.read_excel(file_path_1)
    kmers_frame_reset_ = kmers_frame.reset_index()
    kmers_frame_reset = kmers_frame_reset_.rename(columns={"level_0": "name"})
    name_list = kmers_frame_reset["name"].values.tolist()
    file_path_2 = "D:/work files/gisaid_dataset/"+file_name+"/meta.xlsx"
    meta_tsv= pd.read_excel(file_path_2)
    accession_id_list = []
    collection_time_list = []
    nextstrain_clade_list = []
    for row in name_list:
        row = str(row).replace("[","").replace("]","").replace("'","")
        clade = str(nextclade_tsv.loc[(nextclade_tsv["seqName"]==row)]["clade"].tolist()).replace("'","").replace("[","").replace("]","")
        accession_id = str(meta_tsv.loc[(meta_tsv["strain"]==row)]["gisaid_epi_isl"].tolist()).replace("'","").replace("[","").replace("]","")
        collection_time = str(meta_tsv.loc[(meta_tsv["strain"]==row)]["date"].tolist()).replace("'","").replace("[","").replace("]","")
        accession_id_list.append(accession_id)
        collection_time_list.append(collection_time)
        nextstrain_clade_list.append(clade)
    kmers_frame_reset.insert(1, "accession_id", accession_id_list)
    kmers_frame_reset.insert(2, "collection_time", collection_time_list)
    kmers_frame_reset.insert(3, "nextstrain_clade", nextstrain_clade_list)
    info_kmers_frame = kmers_frame_reset
    return info_kmers_frame


if __name__ == "__main__":
    file_list = ["Argentina","Portugal"]
    for i in file_list:
        kmers_frame = kmers_main(5,i)
        if i == "China":
            info_kmers_frame = frame_clear_1(kmers_frame,i)
        else:
            info_kmers_frame = frame_clear_2(kmers_frame, i)
        file_path = "./"+i+"_info_kmers.xlsx"
        info_kmers_frame.to_excel(file_path,index=False)