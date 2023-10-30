# Anomaly-detection-models-for-SARS-CoV-2-surveillance-based-on-genome-k-mers
## Overview
This project used multiple anomaly detection models for early warning surveillance of new critical variants in SARS-CoV-2 samples based on genome k-mers.
## Introduction of each folder
*data_preprocess*: The scripts in this folder contain sequence data cleaning, k-mers calculations, and sequence lineage labeling.

*data_statistic*: The scripts in this folder include sequence sample statistics for China, Argentina, and Portugal.

*divide_dataset*: The scripts in this folder contain sequence data integrating and dataset dividing. Please refer to our paper for specific dataset dividing methods.

*models*: The scripts in this folder contain 6 single anomaly detection model and 36 stacking models which are consist of these single anomaly detection models. Test these models using the prepared datasets. Please refer to our paper for specific test methods.

*test*: The scripts in this folder using 5 models to detect the new critical variants on the day they first appeared in three countries. "knn_and_lunar.py" includes two models which are KNN and LUNAR. And other scripts represents stacking models.

*days_test*: The scripts in this folder contain dynamic monitoring.
## Analysis pipeline
1. Get virus genome samples and their metadata. Run the codes in *data_preprocess/deal_voci.py* to conduct a dataframe that includes clade information of samples. Run the codes in *data_preprocess/prepare_all_kmers.py* to calculate the k-mers of samples. Then run the codes in *divide_dataset/n_p_dataset_divide.py* and *divide_dataset/time_range_divide.py* to divide all samples into several dataset in order to meet the requirement of tests.
2. Use the datasets prepared to input the models in *models*.
3. Collate the dates when critical variants first appeared in each country's samples. Run the codes in *test* to evaluate the detection ability of five models.
4. Organize the date files required for dynamic monitoring, including country and date information. Then input them into the scripts in *days_test* to do the dynamic monitoring.
