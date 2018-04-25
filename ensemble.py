'''
emble CNN, RNN and RCNN for sleep stage classification
'''
import numpy as np
from numpy.random import seed
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, Average
from tensorflow.python.keras import utils
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard


#hyperparameters
feature_file = "dataset/shhs1-rawFeature.npz"
label_file = "dataset/shhs1-labels.npz"
cnn_model = 'weights/sleepnet-cnn-inter-subject.hdf5'
rnn_model = 'weights/sleepnet-rnn-inter-subject.hdf5'
rcnn_model = 'weights/sleepnet-rcnn-inter-subject.hdf5'

N_samples = 3750 # number of samples in one 30 second epoch
N_channels = 2 # number of EEG channels used
num_channels = 2
FreqSample = 125
lookback = 10
batch_size = 128
step = 2 #
num_patient_per_block = 50

raw_labels = np.load(label_file)
raw_features = np.load(feature_file)
nsrrids = raw_features.keys()

def generator(lookback, min_index, max_index, batch_size=128, step=10):
    '''
    generator which yields timeseries samples and their labels on the fly
    input:
        lookback: how many epochs back the input data should go
        min_index (max_index): indices of the edf files which are used to generate samples
        shuffle: whether to shuffle the samples or draw them in chronological order
        batch_size: the number of samples per batch
        step: the period, in timesteps, at which you sample feature array. Originially, the sampling rate is 125Hz
    output:
        batch_sample: (batch_size, lookback, N_channels*N_samples//step)
        batch_label: (batch_size, 5)
    '''
    if max_index == None:
        max_index = len(nsrrids)
    start_subject_index = min_index
    while 1:
        samples = []
        labels = []
        #read the data from randomly selected num_patient_per_block subjects
        selected_index_nsrrid = np.arange(start_subject_index, min(max_index, start_subject_index+num_patient_per_block))
        for index_nsrrid in selected_index_nsrrid:
            nsrrid = nsrrids[index_nsrrid]
            eeg_raw = raw_features[nsrrid]
            # check the shape of eeg_raw
            k, M, N = eeg_raw.shape
            if k == N_channels and N == 30*FreqSample:
                for i in range(lookback, M+1):
                    if raw_labels[nsrrid][i-1] > 4:
                        continue
                    samples.append(np.swapaxes(eeg_raw[:,i-lookback:i,::step],0,1).reshape(lookback,N_channels*N_samples//step))
                    labels.append(raw_labels[nsrrid][i-1])

        #yeild samples and labels
        samples = np.array(samples)
        labels = np.array(labels)
        #labels = utils.to_categorical(labels, num_classes=5)

        #samples, labels = shuffle(samples, labels) 
        num_sample = samples.shape[0]
        indexes = np.arange(num_sample)
        #np.random.shuffle(indexes)
        for i in range(0,num_sample,batch_size):
            if i+batch_size > num_sample:
                break
            batch_sample = samples[indexes[i:i+batch_size],:,:]
            batch_label = labels[indexes[i:i+batch_size],]
            yield batch_sample, batch_label
        start_subject_index += num_patient_per_block
        if start_subject_index > max_index:
            break
            start_subject_index = min_index

test_gen = generator(lookback, min_index=5201, max_index=None, batch_size=batch_size, step=step)

def plotLoss(history):
    '''
    help function to plot the training loss and validation loss changes versus epochs
    '''
    # visualize error/acc with epochs
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    print (train_loss)
    print (val_loss)
    print (train_acc)
    print (val_acc)

# define cnn model
conv = Sequential(name='cnn')
conv.add(layers.Conv2D(64, (1, 3), activation = 'relu', input_shape = (1, N_channels*30*FreqSample//step, 1)))
conv.add(layers.MaxPooling2D((1, 2)))
conv.add(layers.Conv2D(128, (1, 3), activation = 'relu'))
conv.add(layers.MaxPooling2D((1, 2)))
conv.add(layers.Conv2D(256, (1, 3), activation = 'relu'))
conv.add(layers.MaxPooling2D((1, 2)))
conv.add(layers.Flatten())
conv.add(layers.Dense(64, activation = 'relu'))
conv.add(layers.Dropout(0.5))
conv.add(layers.Dense(5, activation = 'softmax'))
conv.summary()

#define rnn model
rnn = Sequential(name='rnn')
rnn.add(layers.BatchNormalization(input_shape=(None, num_channels*30*FreqSample/step)))
rnn.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1,return_sequences=True))
rnn.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1,return_sequences=True))
rnn.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1,return_sequences=True))
rnn.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1,return_sequences=True))
rnn.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1))
rnn.add(layers.Dense(5, activation="softmax"))
rnn.summary()

#define rcnn model
rcnn = Sequential(name='rcnn')
rcnn.add(layers.BatchNormalization(input_shape=(None, num_channels*30*FreqSample/step)))
rcnn.add(layers.Conv1D(32, 3, activation='relu'))
rcnn.add(layers.MaxPooling1D(2))
rcnn.add(layers.Conv1D(64, 3, activation='relu'))
rcnn.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1,return_sequences=True))
rcnn.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1,return_sequences=True))
rcnn.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1,return_sequences=True))
rcnn.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1))
rcnn.add(layers.Dense(5, activation="softmax"))
rcnn.summary()

conv.load_weights(cnn_model)
rnn.load_weights(rnn_model)
rcnn.load_weights(rcnn_model)


'''
unweighted average
'''
num_batches = 0
num_samples = 0
num_correct_pred_aver = 0
num_correct_pred_vote = 0
for batch_sample, batch_label in test_gen:
    num_batches += 1
    if num_batches < 0:
        break
    if num_batches % 10 == 0:
        print ("unweighted average acc=%f, majority voting acc=%f" % (num_correct_pred_aver / float(num_samples),num_correct_pred_vote / float(num_samples)))
    cnn_sample = batch_sample[:,-1,:].reshape((batch_size,1,-1,1))
    cnn_predict = conv.predict(cnn_sample, batch_size = batch_size)
    rnn_predict = rnn.predict(batch_sample, batch_size = batch_size)
    rcnn_predict = rcnn.predict(batch_sample, batch_size = batch_size)

    #unweighted average
    aver_predict = (cnn_predict+rnn_predict+rcnn_predict)/3.0
    predict = np.argmax(aver_predict, axis=1)
    #predict = np.expand_dims(predict, axis=1)
    num_correct_pred_aver += np.sum(np.equal(predict, batch_label))
    num_samples += batch_size

    #majority voting
    cnn_vote = np.argmax(cnn_predict, axis=1).reshape(1,-1)
    rnn_vote = np.argmax(rnn_predict, axis=1).reshape(1,-1)
    rcnn_vote = np.argmax(rcnn_predict, axis=1).reshape(1,-1)
    vote = np.concatenate((cnn_vote,rnn_vote,rcnn_vote),axis=0)
    vote = [np.bincount(vote[:,i]).argmax() for i in range(vote.shape[1]) ]
    vote = np.array(vote)
    #predict = np.expand_dims(vote, axis=1)
    num_correct_pred_vote += np.sum(np.equal(vote, batch_label))


raw_labels.close()
raw_features.close()
print (num_correct_pred_aver / float(num_samples))
print (num_correct_pred_vote / float(num_samples))
