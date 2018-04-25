'''
RNN for inter-subject sleep stage classification
'''
import numpy as np
from numpy.random import seed
from tensorflow.python.keras import layers, optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras import utils
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.optimizers import Adam

#hyperparameters
feature_file = "dataset/shhs1-rawFeature.npz"
label_file = "dataset/shhs1-labels.npz"
model_file = 'weights/sleepnet-rnn-inter-subject.hdf5' # the name of the file storing the trained cnn model



N_samples = 3750 # number of samples in one 30 second epoch
N_channels = 2 # number of EEG channels used
num_channels = 2
FreqSample = 125
lookback = 10
batch_size = 128
step = 2 #
num_patient_per_block = 70

raw_labels = np.load(label_file)
raw_features = np.load(feature_file)

nsrrids = raw_features.keys()


def generatorWithShuffle(lookback, min_index, max_index, batch_size=128, step=10):
    '''
    generator which yields timeseries samples and their labels on the fly
    input:
        dir_path: the path of the directory which contains all edf files, where features[nsrrid].shape=(k,N,M), k (=1,2) is the number of channels
                  N is the number of epochs for one patient, M is the number of features for one epoch
        labels: the labels for each feature vectors, where labels[nsrrid].shape=(N,)
        lookback: how many epochs back the input data should go
        min_index (max_index): indices of the edf files which are used to generate samples
        shuffle: whether to shuffle the samples or draw them in chronological order
        batch_size: the number of samples per batch
        step: the period, in timesteps, at which you sample feature array. Originially, the sampling rate is 125Hz
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
        labels = utils.to_categorical(labels, num_classes=5)

        #samples, labels = shuffle(samples, labels) 
        num_sample = samples.shape[0]
        indexes = np.arange(num_sample)
        np.random.shuffle(indexes)
        for i in range(0,num_sample,batch_size):
            if i+batch_size > num_sample:
                break
            batch_sample = samples[indexes[i:i+batch_size],:,:]
            batch_label = labels[indexes[i:i+batch_size],]
            yield batch_sample, batch_label
        start_subject_index += num_patient_per_block
        if start_subject_index > max_index:
            start_subject_index = min_index

train_gen = generatorWithShuffle(lookback, min_index=0, max_index=5000, batch_size=batch_size,step=step)
val_gen = generatorWithShuffle(lookback, min_index=5001, max_index=5200, batch_size=batch_size, step=step)

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


model = Sequential(name='rnn')
model.add(layers.BatchNormalization(input_shape=(None, num_channels*30*FreqSample/step)))
model.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1,return_sequences=True))
model.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1,return_sequences=True))
model.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1,return_sequences=True))
model.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1,return_sequences=True))
model.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1))
model.add(layers.Dense(5, activation="softmax"))
model.summary()
#model.load_weights('weights/sleepnet-rnn-500units-epoch20.hdf5')
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.001, decay=0.5), metrics=['accuracy'])

#filepath = 'weights/sleepnet-' + model.name + '-epoch20.hdf5'
checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=0, save_weights_only=True,
                                                 save_best_only=True, mode='auto', period=1)
tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)

history = model.fit_generator(train_gen, 
          steps_per_epoch=1000, 
          epochs=20, 
          validation_data=val_gen, 
          validation_steps = 200,
          callbacks=[checkpoint, tensor_board])

raw_labels.close()
raw_features.close()

plotLoss(history)



