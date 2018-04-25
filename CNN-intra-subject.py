'''
CNN for intra-subject sleep stage annotation 
'''
import numpy as np
from numpy.random import seed
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
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


def generateData(feature_file, label_file, step):
    '''
    generate data for training and testing
    input:
        feature_file: the npz file contains raw features
        label_file: the npz file contains labels
    output:
        samples: numpy array in shape (Nsample, 1, N_channels*N_samples//step, 1)
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

def plotLoss(history):
    '''
    help function to plot the training loss and validation loss changes versus epochs
    ''' 
    # visualize error/acc with epochs
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    print train_loss
    print val_loss
    print train_acc
    print val_acc

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


samples, labels = generateData(feature_file, label_file, step)
print "samples' shape is:"
print samples.shape
print "labels' shape is:"
print labels.shape

X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size = 0.2, random_state = 0)


scaler = StandardScaler()
X_train_ss = scaler.fit_transform(X_train)
X_test_ss = scaler.transform(X_test)

# reshape to be suitable for Keras
X_train_CNN = X_train_ss.reshape((X_train_ss.shape[0], 1, X_train_ss.shape[1], 1)) # shape ((867, 1, 7500, 1))
y_train_CNN = utils.to_categorical(y_train) # shape (867, 6)

X_test_CNN = X_test_ss.reshape((X_test_ss.shape[0], 1, X_test_ss.shape[1], 1)) # shape ((217, 1, 7500, 1))
y_test_CNN = y_test
y_test_CNN = utils.to_categorical(y_test) # shape (217, 6)

print (X_train_CNN.shape, X_test_CNN.shape, y_train_CNN.shape, y_test_CNN.shape)

conv = Sequential()
conv.add(Conv2D(64, (1, 3), activation = 'relu', input_shape = (1, N_channels*30*FreqSample//step, 1)))
conv.add(MaxPooling2D((1, 2)))

conv.add(Conv2D(128, (1, 3), activation = 'relu'))
conv.add(MaxPooling2D((1, 2)))

conv.add(Conv2D(256, (1, 3), activation = 'relu'))
conv.add(MaxPooling2D((1, 2)))

conv.add(Flatten())
conv.add(Dense(64, activation = 'relu'))
conv.add(Dropout(0.5))
conv.add(Dense(5, activation = 'softmax'))

conv.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

history = conv.fit(X_train_CNN, y_train_CNN, batch_size = 128, epochs = 10, validation_data = (X_test_CNN, y_test_CNN))
plotLoss(history)


