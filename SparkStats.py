'''
use spark to extract descriptive statistics of the dataset 
'''
import pyspark
from pyspark import SparkContext
import numpy as np
import pyspark
import sklearn.datasets
import pandas as pd

N_samples = 3750 # number of samples in one 30 second epoch
N_channels = 2 # number of EEG channels used
FreqSample = 125
step = 2 #


def DesStats(feature_file, label_file):
    '''
    use spark to extract descriptive statistics of the features and labels
    '''
    #load raw features and labels as numpy array
    raw_labels = np.load(label_file)
    raw_features = np.load(feature_file)
    nsrrids = raw_features.keys()
    np_data = []
    for nsrrid in nsrrids:
        if int(nsrrid) > 200005:
            continue
        eeg_raw = raw_features[nsrrid]
        # check the shape of eeg_raw
        k, M, N = eeg_raw.shape
        if k == N_channels or N == 30*FreqSample:
            for i in range(M):
                if raw_labels[nsrrid][i] > 4:
                    continue
                np_data.append(np.append(eeg_raw[:,i,::step].reshape(-1),int(raw_labels[nsrrid][i])))
    samples = np.array(np_data)
    print (samples.shape)
    raw_labels.close()
    raw_features.close()
 
    # create data frame based on the loaded numpy array
    sc = pyspark.SparkContext.getOrCreate()
    session = pyspark.sql.SparkSession(sc)
    #data_pd = pd.DataFrame(samples)
 
    
    data = session.createDataFrame([tuple([float(samples[i][j]) for j in range(samples[i].shape[0])]) for i in range(samples.shape[0])],
                                   ['f'+str(i) for i in range(samples.shape[0]-1)].append('label'))
    
    #data = session.createDataFrame(data_pd)
    #data.printSchema()
    # get the distribution of labels
    data.createOrReplaceTempView("eeg")
    session.sql("select count(label) from eeg group by label").show()
    data.groupBy("label").count().show()

if __name__ == '__main__':
    
    feature_name = 'dataset/shhs1-rawFeature.npz'
    label_name = 'dataset/shhs1-labels.npz'
    DesStats(feature_name, label_name)
