'''
RNN for intra-subject sleep stage classification
'''
import numpy as np
from numpy.random import seed
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras import utils
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

#hyperparameters
feature_file = "dataset/shhs1-rawFeature.npz"
label_file = "dataset/shhs1-labels.npz"

N_samples = 3750 # number of samples in one 30 second epoch
N_channels = 2 # number of EEG channels used
num_channels = 2
FreqSample = 125
lookback = 10
batch_size = 128
step = 2 #


def generateData(feature_file, label_file, lookback, step):
    '''
    generate data for training and testing
    input:
        feature_file: the npz file contains raw features
        label_file: the npz file contains labels
    output:
        samples: numpy array in shape (Nsample, lookback, N_channels*N_samples//step)
        lables: numpy array in shape (Nsample, )
    '''

    #read data
    raw_labels = np.load(label_file)
    raw_features = np.load(feature_file)
    nsrrids = raw_features.keys()
    samples = []
    labels = []
    #samples = np.zeros((1,1,N_channels*N_samples//step,1))
    #labels = np.zeros((1,))
    #labels[0] = 4 # make sure that all possible classes are presented
    #num_file = 0
    for nsrrid in nsrrids:
        if int(nsrrid) > 200100:
            continue       
        '''
        if num_file > 100:
            break
        num_file += 1
        '''
        print ("reading nsrrid=%s" % nsrrid)
        eeg_raw = raw_features[nsrrid]
        # check the shape of eeg_raw
        k, M, N = eeg_raw.shape
        if k != N_channels:
            continue
            #raise NameError('wrong number of channels, found %s, should be %s' % (N_channels, k))
        if N != 30*FreqSample:
            raise NameError('wrong sampling frequency, found %s, should be %s' % (FreqSample, N/30))
        for i in range(lookback, M):
            if raw_labels[nsrrid][i-1] > 4:
                continue
            samples.append(np.swapaxes(eeg_raw[:,i-lookback:i,::step],0,1).reshape(lookback,N_channels*N_samples//step))
            labels.append(raw_labels[nsrrid][i-1])
    samples = np.array(samples)
    labels = np.array(labels)
    #labels = utils.to_categorical(labels)
    raw_labels.close()
    raw_features.close()
    return samples, labels

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

    epochs = range(1, len(val_acc) + 1)
    fig = plt.figure(figsize = (10, 6), dpi = 100)
    #plot of train/val loss
    fig.add_subplot(211)
    plt.plot(epochs, train_loss, 'bo', label = 'training loss')
    plt.plot(epochs, val_loss, 'b-', label = 'validation loss')
    plt.title('Training/validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)
    plt.legend(loc=0)
    # plot of train/val acc
    fig.add_subplot(212)
    plt.plot(epochs, train_acc, 'ro', label = 'training acc')
    plt.plot(epochs, val_acc, 'r-', label = 'validation acc')
    plt.title('Training/validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)
    plt.legend(loc=0)
    fig.tight_layout()
    plt.show()   


samples, labels = generateData(feature_file, label_file, lookback, step)
print ("samples' shape is:")
print (samples.shape)
print ("labels' shape is:")
print (labels.shape)

X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size = 0.2, random_state = 0)

mean_train = X_train[:,-1,:].mean(axis=0)
X_train -= mean_train
std_train = X_train[:,-1,:].std(axis=0)
X_train /= std_train

X_test -= mean_train
X_test /= std_train

# reshape to be suitable for Keras
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = Sequential()
model.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1,input_shape=(None,num_channels*30*FreqSample/step),return_sequences=True))
model.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1,return_sequences=True))
model.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1,return_sequences=True))
model.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1,return_sequences=True))
model.add(layers.LSTM(1000,dropout=0.1, recurrent_dropout=0.1))
model.add(layers.Dense(5, activation="softmax"))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size = 128, epochs = 10, validation_data = (X_test, y_test))
plotLoss(history)


