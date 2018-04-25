'''
This script is used to extract features from the raw EEG signals based on S. Biswal et al. SleepNet: Automated Sleep Staging System via Deep Learning
All features will be stored in a npz file, and all labels will be in another npz file
You need to specify the paths to edf files and xml files, you also need to set the paths of the generated npz files
'''
import numpy as np
import mne
import sys, os, re
import xml.etree.ElementTree as ET
from scipy.sparse import csr_matrix
import tempfile

############################
# help function, enable array saving one by one
############################
class my_savez(object):
    def __init__(self, file):
        # Import is postponed to here since zipfile depends on gzip, an optional
        # component of the so-called standard library.
        import zipfile
        # Import deferred for startup time improvement
        import tempfile
        import os

        if isinstance(file, basestring):
            if not file.endswith('.npz'):
                file = file + '.npz'

        compression = zipfile.ZIP_STORED

        zip = self.zipfile_factory(file, mode="w", compression=compression)

        # Stage arrays in a temporary file on disk, before writing to zip.
        fd, tmpfile = tempfile.mkstemp(suffix='-numpy.npy')
        os.close(fd)

        self.tmpfile = tmpfile
        self.zip = zip
        self.i = 0

    def zipfile_factory(self, *args, **kwargs):
        import zipfile
        import sys
        if sys.version_info >= (2, 5):
            kwargs['allowZip64'] = True
        return zipfile.ZipFile(*args, **kwargs)

    def savez(self, *args, **kwds):
        import os
        import numpy.lib.format as format

        namedict = kwds
        for val in args:
            key = 'arr_%d' % self.i
            if key in namedict.keys():
                raise ValueError("Cannot use un-named variables and keyword %s" % key)
            namedict[key] = val
            self.i += 1

        try:
            for key, val in namedict.iteritems():
                fname = key + '.npy'
                fid = open(self.tmpfile, 'wb')
                try:
                    format.write_array(fid, np.asanyarray(val))
                    fid.close()
                    fid = None
                    self.zip.write(self.tmpfile, arcname=fname)
                finally:
                    if fid:
                        fid.close()
        finally:
            os.remove(self.tmpfile)

    def close(self):
        self.zip.close()

##################################
# step 1: read data from edf files
##################################
def read_edf_file(file_path):
    '''
    read a raw edf file, and get the sampling data for the given channels
    input:
        file_path: the path to the edf file
    output:
        eeg_raw: a k*N shape numpy array, where k (k=2) is the number of channels, N=Fs*T is the total number of samples for one patient, Fs is the sampling rate, T is the recording duration
        Fs: sampling rate
    '''
    # read the whole edf file
    if not os.path.isfile(file_path):
        raise Exception ("%s is not found" % file_path)
    EDF = mne.io.read_raw_edf(file_path)
    channels = EDF.info['ch_names']
    eeg_raw = EDF.to_data_frame()
    eeg_raw = eeg_raw.as_matrix().T
    
    # find the data for the given channels
    channel_id = []
    for index, name in enumerate(channels):
        if 'EEG' in name:
            channel_id.append(index)
    eeg_raw = eeg_raw[channel_id,:]
    return eeg_raw, int(EDF.info['sfreq'])

##################################
# step 2: extract features
#################################
def extractRawFeature(dir_path, output_file):
    '''
    extract the raw eeg feature for each 30 second epoch, and output the generated features to a text file
    The feature array outputed will be in shape of k*n*30Fs, where k is the number of channels, Fs is the samping rate, and n is the number of epoches. The feature arrays for all patients will be outputted to a npz file
    input:
        dir_path: the full path of the directory which contains all edf files
        output_file: the path of the output file which stores all the features
    output:
        None
    '''
    #result = []
    #tmp = tempfile.TemporaryFile()
    f = my_savez(output_file)
    num_files = 0
    for fname in os.listdir(dir_path):
        if num_files % 100 == 0:
            print "extracting %d files for raw features" % num_files
        '''     
        if num_files == 5:
            break
        '''
        nsrrid = fname.split('-')[1].split('.')[0]
        '''
        if int(nsrrid) > 200100:
            continue
        '''
        fname = dir_path+'/'+fname
        eeg_raw, Fs = read_edf_file(fname)
        k, M = eeg_raw.shape
        eeg_raw = eeg_raw.reshape((k, M/(30*Fs), 30*Fs))
        #result.append((nsrrid, eeg_raw))
        f.savez(**dict([(nsrrid, eeg_raw)]))
        num_files += 1
    f.close()
    #np.savez(output_file, **dict(result))

#################################
# step 3: extract labels
#################################
def extractLabels(dir_path,output_file):
    '''
    extract labels for all eeg epoches from given xml files, and output the labels to text file
    The output matrix is in shape of N*n, where N is the number of patient, n is the number of epoches.
    input:
        dir_path: the directory which contains the xml files
        output_file: the outputted file
    output:
        None
    '''
    result = []
    row_indexer = 0
    num_files = 0
    for filename in os.listdir(dir_path):
        if num_files % 100 == 0:
            print "processing %d files for labels" % num_files
        num_files += 1
        '''
        if num_files == 5:
            break
        '''
        nsrrid = filename.split('-')[1]
        filename = dir_path+'/'+filename
        tree = ET.parse(filename)
        root = tree.getroot()
        data = []
        for scoredEvent in root.iterfind('ScoredEvents/ScoredEvent'):
            eventType = scoredEvent.find('EventType').text
            if eventType == "Stages|Stages":
                stage = int(scoredEvent.find('EventConcept').text.split('|')[1])
                start = int(float(scoredEvent.find('Start').text))
                duration = int(float(scoredEvent.find('Duration').text))
                for i in range(duration/30):
                    data.append(stage) # +1 can make the value of stage >0, which is helpful for distinguish 0 in sparse matrix
        data = np.asarray(data)
        result.append((nsrrid,data))
    np.savez(output_file,**dict(result))


################################
# step 4: extract spectrogram features
################################
def extractSpectrogramFeature():
    return

################################
# extract raw features and labels 
###############################
extractRawFeature("edfs/shhs1","dataset/shhs1-rawFeature")
extractLabels("annotations-events-nsrr/shhs1","dataset/shhs1-labels")


