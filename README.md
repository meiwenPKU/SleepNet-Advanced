# SleepNet-Advanced

This repository is created by Hemin Yang (hyang350@gatech.edu) and Chenyang Shi (cs3000@columbia.edu) for the final project in Gatech cse 6250 2018 spring, containing the source codes which try to reproduce the results reported in [SleepNet](https://arxiv.org/pdf/1707.08262.pdf). However, we use a different dataset ([SHHS](https://sleepdata.org/datasets/shhs)) because the dataset used in [SleepNet](https://arxiv.org/pdf/1707.08262.pdf) is not publicaly available

Please follow the belowing steps to reproduce our results

# Step 1: download the orgininal data
1.1) download the annotation xml files for shhs1 from https://sleepdata.org/datasets/shhs/files/polysomnography/annotations-events-nsrr/shhs1

1.2) download the edf files for shhs1 from https://sleepdata.org/datasets/shhs/files/polysomnography/edfs/shhs1 

# Step 2: generate raw data
By running [genRaw.py](genRaw.py), generate raw features and labels. All features will be stored in shhs1-rawFeature.npz, and labels in shhs1-labels.npz

# Step 3: Collect descriptive statistics 
Collect descriptive statistics of the features and labels with PySpark by running [SparkStats.py](SparkStats.py)

# Step 4: Intra-subject experiments
## Basic machine learning algorithms
Use grid search provided by sklearn to tune the hyperparameters for different classification algorithms in intra-subject annotation by running [BasicML.py](BasicML.py)
## CNN
Use CNN model for intra-subject training and testing by running [CNN-intra-subject.py](CNN-intra-subject.py)
## RNN
Use RNN model for intra-subject training and testing by running [RNN-intra-subject.py](RNN-intra-subject.py)
## RCNN
Use RCNN model for intra-subject training and testing by running [RCNN-intra-subject.py](RCNN-intra-subject.py)
# Step 5: Inter-subject experiments
## CNN
Use CNN model for inter-subject training and validation by running [CNN-inter-subject.py](CNN-inter-subject.py), for testing by running [CNN-inter-subject-test.py](CNN-inter-subject-test.py)
## RNN
Use RNN model for inter-subject training and testing by running [RNN-inter-subject.py](RNN-inter-subject.py), for testing by running [RNN-inter-subject-test.py](RNN-inter-subject-test.py)
## RCNN
Use RCNN model for inter-subject training and testing by running [RCNN-inter-subject.py](RCNN-inter-subject.py), for testing by running [RCNN-inter-subject-test.py](RCNN-inter-subject-test.py)
## Unweighted average and majority voting
use unweighted average and majority voting to ensemble the trained CNN, RNN, and RCNN models and evaluate the performance of ensembled model by running [ensemble.py](ensemble.py)

## Super Learner
Use super learner to ensemble the trained CNN, RNN, and RCNN models and evaluate the performance of ensembled model by running [superLearner-main.py](superLearner-main.py)
