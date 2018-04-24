# SleepNet-Advanced

This repository contains the source codes which try to reproduce the results reported in [SleepNet](https://arxiv.org/pdf/1707.08262.pdf). However, a different dataset ([SHHS](https://sleepdata.org/datasets/shhs)) will be used because the dataset used in [SleepNet](https://arxiv.org/pdf/1707.08262.pdf) is not publicaly available
Please follow the following steps to reproduce the results

# Step 1: download the orgininal data
1.1) download the annotation xml files for shhs1 from https://sleepdata.org/datasets/shhs/files/polysomnography/annotations-events-nsrr/shhs1

1.2) download the edf files for shhs1 from https://sleepdata.org/datasets/shhs/files/polysomnography/edfs/shhs1 

# Step 2: generate raw data
By running genRaw.py, generate raw features and labels. All features will be stored in shhs1-rawFeature.npz, and labels in shhs1-labels.npz

# Step 3: Descriptive statistics 

# Step 4: Intra-subject experiments
## Basic machine learning algorithms

## CNN

## RNN

## RCNN

# Step 5: Inter-subject experiments
## CNN

## RNN

## RCNN

## Unweighted average

## Majoirty vote

## Super Learner
