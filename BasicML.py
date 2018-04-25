'''
Using grid search provided by sklearn to tune the hyperparameters for different classification algorithms
You need to specify the range of each hyperparameter
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFECV
from sklearn.externals import joblib
from datetime import datetime
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import os
from collections import Counter
from sklearn import preprocessing


#hyperparameters
feature_file = "dataset/shhs1-rawFeature.npz" # provide the right path to the feature npz file and lable npz file
label_file = "dataset/shhs1-labels.npz"

N_samples = 3750 # number of samples in one 30 second epoch
N_channels = 2 # number of EEG channels used
num_channels = 2
FreqSample = 125
lookback = 10
batch_size = 128
step = 2 #

classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    LogisticRegression()]
names = ["Linear SVM", "RBF/Poly SVM","Decision Tree", "Random Forest", "Neural Net", "AdaBoost","LR"]

tuned_parameters=[
    {'clf__kernel':['linear'],'clf__C':[1e-1,1e0,1e1]},
    {'clf__kernel':['rbf'],'clf__C':[1e0,1e1],'clf__gamma':[1e0]},
    {'clf__criterion':['gini'],'clf__max_depth':[5,8]},
    {'clf__criterion':['gini'],'clf__max_depth':[50],'clf__n_estimators':[10,20,30,50]},
    {'clf__hidden_layer_sizes':[(30),(15,15),(10,10,10)],'clf__alpha':[1e-1, 1e0],'clf__learning_rate': ['invscaling'],'clf__learning_rate_init':[1e-2]},
    {'clf__n_estimators':[20,40,60,100,200],'clf__learning_rate':[0.01, 0.1, 1]},
    {'clf__C':[0.01, 0.1, 1, 10]}]
    #{'clf__loss':['deviance','exponential'],'clf__learning_rate':[0.01], 'clf__n_estimators':[60],'clf__max_depth':[15,20,25,30]}]


def generateData(feature_file, label_file, step):
    '''
    generate data for training and testing
    input:
        feature_file: the npz file contains raw features
        label_file: the npz file contains labels
    output:
        samples: numpy array in shape (Nsample, N_channels*N_samples//step)
        lables: numpy array in shape (Nsample, )
    '''

    #read data
    raw_labels = np.load(label_file)
    raw_features = np.load(feature_file)
    nsrrids = raw_features.keys()
    samples = []
    labels = []
    for nsrrid in nsrrids:
        if int(nsrrid) > 200100:
            continue
        print ("reading nsrrid=%s" % nsrrid)
        eeg_raw = raw_features[nsrrid]
        # check the shape of eeg_raw
        k, M, N = eeg_raw.shape
        if k != N_channels:
            raise NameError('wrong number of channels, found %s, should be %s' % (N_channels, k))
        if N != 30*FreqSample:
            raise NameError('wrong sampling frequency, found %s, should be %s' % (FreqSample, N/30))
        for i in range(M):
            if raw_labels[nsrrid][i] > 4:
                continue
            samples.append(eeg_raw[:,i,::step].reshape(-1))
            labels.append(raw_labels[nsrrid][i])
    samples = np.array(samples)
    labels = np.array(labels)
    #labels = utils.to_categorical(labels)
    raw_labels.close()
    raw_features.close()
    return samples, labels

samples, labels = generateData(feature_file, label_file, step)
X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size = 0.2, random_state = 0)

print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

for parameters, clf, name in zip(tuned_parameters, classifiers,names):
    
    if  "SVM" in name or name == 'AdaBoost':
        continue
    
    print ("cv on classifier %s" % (name))
    estimators = [('standardization', StandardScaler()),('clf',clf)]
    pipe = Pipeline(estimators)
    clf_gs = GridSearchCV(pipe, param_grid=parameters, cv=5, n_jobs=10, verbose=3, scoring='accuracy')
    clf_gs.fit(X_train, y_train)

    means = clf_gs.cv_results_['mean_test_score']
    stds = clf_gs.cv_results_['std_test_score']
    score_times = clf_gs.cv_results_['mean_score_time']
    for mean, std, scoretime, params in zip(means, stds, score_times, clf_gs.cv_results_['params']):
        #print ("%0.3f (+/-%0.03f), %s" % (mean, std * 2, scoretime))
        print("%0.3f (+/-%0.03f), score time=%s, for %r" % (mean, std * 2, scoretime, params))

