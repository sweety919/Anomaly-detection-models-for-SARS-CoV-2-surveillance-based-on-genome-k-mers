# Evaluation-of-anomaly-detection-models-for-SARS-CoV-2-surveillance-based-on-genome-k-mers
## Overview
This project used multiple anomaly detection models for early warning surveillance of new critical variants in SARS-CoV-2 samples.
## Introduction of each folder
*data_preprocess*: The scripts in this folder contain sequence data cleaning, k-mers calculations, and sequence lineage labeling.

*data_statistic*: The scripts in this folder include sequence sample statistics for China, Argentina, and Portugal.

*divide_dataset*: The scripts in this folder contain sequence data integrating and dataset dividing. Please refer to our paper for specific dataset dividing methods.

*models*: The scripts in this folder contain 6 single anomaly detection model and 36 stacking models which are consist of these single anomaly detection models. Test these models using the prepared datasets. Please refer to our paper for specific test methods.

*test*: The scripts in this folder using 5 models to detect the new critical variants on the day they first appeared in three countries. "knn_and_lunar.py" includes two models which are KNN and LUNAR. And other scripts represents stacking models.

*days_test*: The scripts in this folder contain dynamic monitoring.
