# Original Authors:
# Imam, Nabil  Cleland, Thomas [tac29 at cornell.edu]
# 2020
# https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=261864#tabs-1
# 
# Modified by Nik Dennler, October 2022, n.dennler2@herts.ac.uk
# 
# ATTENTION: Run with Python 2.7!

import os
import csv
import pickle
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

def loadFile(fileName, time_sample):
    """
    Load data points of a given data file and point in time

    :param str fileName: filename
    :param int time_sample: sampling time
    :return list, int: all sensor readings, exact sampling time in seconds
    """
    lines = []
    data = csv.reader(open(fileName, 'r'), delimiter='\t')
    for i in data:
        lines.append(i)

    odor_raw = [] 
    for i in range(0, len(lines)): 
        time = int(float(lines[i][0]))/1000.0
        if time > time_sample:          #load the data recorded at given time
            #odor_raw.append(time)
            for j in range(12, 92):         #gas sensor data is stored at these locations
                if lines[i][j]!='1':
                    odor_raw.append(int(float(lines[i][j])))
            break
            #odor_raw[k].append( round( float(lines[i][j]) *2, 6)/2 )		
    return odor_raw, time 

def find_gas_file(data_dir, gas='CO_1000'):
    """
    In data directory, find gas file that matches a given gas string

    :param str data_dir: data directory
    :param str gas: gas identifying string, defaults to 'CO_1000'
    :return list: data files where string is matched
    """
    data_files = []
    for d in os.listdir(data_dir):
        if d.find(gas) != -1:
            data_files.append(data_dir+"/"+d)
    return data_files

def get_gas_files(data_dir, gas, time_sample, n_samples=10):
    """
    Get n gas files for a given gas and time, plus baseline

    :param str data_dir: data directory
    :param str gas: gas identifying string
    :param int time_sample: sampling time
    :param int n_samples: number of gas samples per gas, defaults to 10
    :return list, list: gas data, baseline data
    """
    time_baseline = 1   # t=1s 

    raw_all = []
    baseline_all = []
    
    filenames = find_gas_file(data_dir, gas=gas)

    # randomly select n_samples
    for i in range(n_samples):
        filename = random.choice(filenames)
        raw, time = loadFile(filename, time_sample=time_sample)
        baseline, baselinetime = loadFile(filename, time_sample=time_baseline)

        raw_all.append(raw)
        baseline_all.append(baseline)
    
    return raw_all, baseline_all


def loadData(time_sample=int('090'), dir="testing", n_samples=10):
    """
    Load training or testing data. Returns raw data, baseline data and labels

    :param int time_sample: sampling time, defaults to int('090')
    :param str dir: data directory name, defaults to "testing"
    :param int n_samples: number of gas samples per gas, defaults to 10
    :return list, list, list: odors_raw, odors_baseline, odors_labels
    """

    data_dir = "data/"+dir

    Toluene_raw, Toluene_baseline = get_gas_files(data_dir, 'Toluene_200', time_sample, n_samples=n_samples)
    Benzene_raw, Benzene_baseline = get_gas_files(data_dir, 'Benzene_200', time_sample, n_samples=n_samples)
    Methane_raw, Methane_baseline = get_gas_files(data_dir, 'Methane_1000', time_sample, n_samples=n_samples)
    CO_raw, CO_baseline = get_gas_files(data_dir, 'CO_1000', time_sample, n_samples=n_samples)
    Ammonia_raw, Ammonia_baseline = get_gas_files(data_dir, 'Ammonia_10000', time_sample, n_samples=n_samples)
    Acetone_raw, Acetone_baseline = get_gas_files(data_dir, 'Acetone_2500', time_sample, n_samples=n_samples)
    Acetaldehyde_raw, Acetaldehyde_baseline = get_gas_files(data_dir, 'Acetaldehyde_500', time_sample, n_samples=n_samples)
    Methanol_raw, Methanol_baseline = get_gas_files(data_dir, 'Methanol_200', time_sample, n_samples=n_samples)    
    Butanol_raw, Butanol_baseline = get_gas_files(data_dir, 'Butanol_100', time_sample, n_samples=n_samples)
    Ethylene_raw, Ethylene_baseline = get_gas_files(data_dir, 'Ethylene_500', time_sample, n_samples=n_samples)       

    odors_labels = ["Toluene", "Benzene", "Methane", "CO", "Ammonia", "Acetone", "Acetaldehyde", "Methanol", "Butanol", "Ethylene"]
    odors_raw = [Toluene_raw, Benzene_raw, Methane_raw, CO_raw, Ammonia_raw, Acetone_raw, Acetaldehyde_raw, Methanol_raw, Butanol_raw, Ethylene_raw]
    odors_baseline = [Toluene_baseline, Benzene_baseline, Methane_baseline, CO_baseline, Ammonia_baseline, Acetone_baseline, Acetaldehyde_baseline, Methanol_baseline, Butanol_baseline, Ethylene_baseline]
    
    odors_labels = [odors_labels[i] for i, sublist in enumerate(odors_raw) for item in sublist] # Flattening
    odors_raw = [item for sublist in odors_raw for item in sublist] # Flattening
    odors_baseline = [item for sublist in odors_baseline for item in sublist] # Flattening

    return odors_raw, odors_baseline, odors_labels


def findDynamicRange(odors_raw):
    """
    Finding dynamic range based on the min-max of each of the 72 sensor, across all gases

    :param list odors_raw: List of raw data
    :return nested list: list of tuples with dynamic range boundaries
    """
    nSensors = len(odors_raw[0])
    dRange = [[0,0]]*nSensors        #(min, max) for each sensor
    for i in range(0, len(odors_raw)):
        for j in range(0, nSensors):  #+1 because 0 is timestamp 
            if(i==0):
                dRange[j] = [odors_raw[i][j], odors_raw[i][j]]
            elif odors_raw[i][j] < dRange[j][0]:     #new < min
                dRange[j][0] = odors_raw[i][j] 
            elif odors_raw[i][j] > dRange[j][1]:     #new > max
                dRange[j][1] = odors_raw[i][j] 
    return dRange 

def findBinSpacing(odors_raw, nBins):
    """
    Finding bin spacing for discretisation of sensor data

    :param list odors_raw: raw sensor data
    :param int nBins: number of bins
    :return nested list, list: dynamic range andbin spacing
    """
    dRange = findDynamicRange(odors_raw)
    binSpacing = []
    for i in dRange:
        interval = i[1]-i[0]
        binSpacing.append(round(interval/float(nBins-1), 4))
    return dRange, binSpacing 

def binData(odorMainUnbinned, binSpacing, dRange, nBins):
    """
    Binned data

    :param list odorMainUnbinned: raw sensor data
    :param list binSpacing: bin spacing
    :param list dRange: dynamic range
    :param int nBins: number of bins
    :return list: binned sensor data
    """
    odorMain = [] 
    for i in range(0, len(odorMainUnbinned)):
        odorMain.append([]) 
        for j in range(0, len(dRange)):
            temp = (odorMainUnbinned[i][j] - dRange[j][0])/binSpacing[j] 
            temp = np.clip(int(round(temp)), 0, nBins-1)
            odorMain[i].append(temp) 
    return odorMain 

def sparsifySingle(odorDense):
    """
    Sparsify sensor array recordings by setting a least-dominant fraction to zero

    :param list odorDense: non-sparsified sensor recordings
    :return list: sparsified sensor recordings
    """
    top = [0]*72           # list of most active sensors
    odorTemp = copy.deepcopy(odorDense) 
    cutoff = 36        # number of sensors that make the top list
    
    for i in range(0, cutoff):
        m = max(odorTemp)
        index1 = odorTemp.index(m) 
        odorTemp[index1] = 0
        top[index1] = m
    return top

def sparsifyOdors(odorsDense):
    """
    For list of sensor array recordings, sparsify recordings

    :param list odorsDense: non-sparsified sensor recordings, for all odours
    :return list: non-sparsified sensor recordings, for all odours
    """
    odorsSparsified = []
    for i in odorsDense:
        s = sparsifySingle(i)
        odorsSparsified.append(s)
    return odorsSparsified

def AddOcclusion(
data = [[]],
n = 1,              # number of samples per noise level
pList = [0.5],      # mean of bernoulli process 
):
    """
    Add random noise to fraction of data

    :param list data: non-occluded data, defaults to [[]]
    :param int n: number of samples per noise level, defaults to 1
    :param list pList: mean of bernoulli process 
    :return list: occluded data
    """
    noisy_data = [] 
    l=-1 
    for i in range(0, len(data)):
        ndim = len(data[i])        # dimensionality of data
        for j in range(0, n):
            for p in pList:
                noisy_data.append([])
                l+=1
                affected_ids = random.sample(range(ndim), int(p*ndim))
                for k in range(0, ndim):
                    if k in affected_ids:
                        noise_act = random.randint(0, 15)          # random destructive interference 
                        noisy_data[l].append(noise_act)
                    else:
                        noisy_data[l].append(data[i][k])
    return noisy_data

def offset_subtraction(raw, baseline):
    """
    Subtract baseline from raw data, which allows a more truthful validation of classification accuracy

    :param list raw: odour recordings
    :param list baseline: baseline recordings
    :return _type_: baseline subtracted odour recordings
    """
    return (np.array(raw)-np.array(baseline)).tolist()

def plot_data(odors_raw_training, odors_raw_testing, trainingOdors, testingOdors, odor_labels_training, odor_labels_testing, name):
    fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True, sharey='col', figsize=(8,5))
    match = np.sum(np.array(trainingOdors[0])==np.array(testingOdors[0]))
    ax[0].scatter(range(len(odors_raw_training[0])), odors_raw_training[0], c='b', alpha=0.5, label=odor_labels_training[0])
    ax[0].scatter(range(len(odors_raw_testing[0])), odors_raw_testing[0], c='r', alpha=0.5, label=odor_labels_testing[0])
    ax[1].scatter(range(len(trainingOdors[0])), trainingOdors[0], c='b', alpha=0.5)
    ax[1].scatter(range(len(testingOdors[0])), testingOdors[0], c='r', alpha=0.5, label="match: "+str(match) +"/72")
    ax[0].set_title("Raw")
    ax[1].set_title("Augmented")
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[0].set_ylabel("Value")
    ax[0].set_xlabel("Channel")
    ax[1].set_xlabel("Channel")
    plt.suptitle("Offset subtraction: " + str(OFFSET_SUBTRACTION) + "\n Occlusion: " + str(NOISE_LEVEL))
    plt.savefig("visualise_data_" + name + ".svg")
    plt.close()

if __name__ == '__main__':

    # Define experiments
    all_experiments = {
        "experiment1" : {
            "OFFSET_SUBTRACTION": False,
            "SEPARATE_TRAIN_TEST": False,
            "TIME_SAMPLE_TRAIN": "090",
            "TIME_SAMPLE_TEST": "090",
            "NOISE_LEVEL": 0.6,
        },
        "experiment2" : {
            "OFFSET_SUBTRACTION": False,
            "SEPARATE_TRAIN_TEST": False,
            "TIME_SAMPLE_TRAIN": "015",
            "TIME_SAMPLE_TEST": "015",
            "NOISE_LEVEL": 0.6,
        },
        "experiment3" : {
            "OFFSET_SUBTRACTION": False,
            "SEPARATE_TRAIN_TEST": True,
            "TIME_SAMPLE_TRAIN": "090",
            "TIME_SAMPLE_TEST": "090",
            "NOISE_LEVEL": 0.6,
        },    
        "experiment4" : {
            "OFFSET_SUBTRACTION": True,
            "SEPARATE_TRAIN_TEST": True,
            "TIME_SAMPLE_TRAIN": "090",
            "TIME_SAMPLE_TEST": "090",
            "NOISE_LEVEL": 0.6,
        },
        "experiment5" : {
            "OFFSET_SUBTRACTION": True,
            "SEPARATE_TRAIN_TEST": True,
            "TIME_SAMPLE_TRAIN": "090",
            "TIME_SAMPLE_TEST": "090",
            "NOISE_LEVEL": 0,
        }, 
    }

    # Iterate over experiments
    for experiment, params in all_experiments.items():
        OFFSET_SUBTRACTION = params["OFFSET_SUBTRACTION"]
        SEPARATE_TRAIN_TEST = params["SEPARATE_TRAIN_TEST"]
        TIME_SAMPLE_TRAIN = params["TIME_SAMPLE_TRAIN"]
        TIME_SAMPLE_TEST = params["TIME_SAMPLE_TEST"]
        NOISE_LEVEL = params["NOISE_LEVEL"]
        SAME_BINS = True
        VISUALISE_DATA = True

        experiment_name = str(NOISE_LEVEL) + "_" + TIME_SAMPLE_TRAIN + "s_" + TIME_SAMPLE_TEST + "s_SO_" + str(OFFSET_SUBTRACTION) + "_controltest" + str(SEPARATE_TRAIN_TEST) + "_samebins" + str(SAME_BINS)
        print(experiment_name)
        random.seed(1)

        # Extract data used in paper
        odors_raw_training, odors_raw_training_baseline, odor_labels_training = loadData(dir="training", time_sample=int(TIME_SAMPLE_TRAIN), n_samples=1)

        # Subtract Offset
        if OFFSET_SUBTRACTION:
            odors_raw_training = offset_subtraction(odors_raw_training, odors_raw_training_baseline)

        #Binning and sparsification
        nBins = 16 
        dRange_train, binSpacing_train = findBinSpacing(odors_raw_training, nBins)
        odorsDense = binData(odors_raw_training, binSpacing_train, dRange_train, nBins)
        odors_training = sparsifyOdors(odorsDense) 
        trainingOdors = []
        for odor in odors_training:
            trainingOdors.append(copy.deepcopy(odor))

        # Testing
        nTest = 10
        noiseLevels = [NOISE_LEVEL]
        
        # Training and Testing on same datapoints
        if not SEPARATE_TRAIN_TEST:
            odors_raw_testing, odors_raw_testing_baseline, odor_labels_testing = loadData(dir="training", time_sample=int(TIME_SAMPLE_TEST), n_samples=1)
            n_occlude = nTest

        # Training and Testing on separate datapoints
        else:
            print("train & test on separate data")
            odors_raw_testing, odors_raw_testing_baseline, odor_labels_testing = loadData(dir="testing", time_sample=int(TIME_SAMPLE_TEST), n_samples=nTest)
            n_occlude = 1

        if OFFSET_SUBTRACTION:
            odors_raw_testing = offset_subtraction(odors_raw_testing, odors_raw_testing_baseline)

        # Binning and sparsification. IMPORTANT: We use same binning for training & testing as in the original paper when plumes are considered. Tried also to re-do binning, but performance drops.
        if SAME_BINS:
            binSpacing = binSpacing_train
            dRange = dRange_train
        else:
            dRange_test, binSpacing_test = findBinSpacing(odors_raw_testing, nBins)
            binSpacing = binSpacing_test
            dRange = dRange_test
        odorsDense = binData(odors_raw_testing, binSpacing, dRange, nBins) 
        odors_testing = sparsifyOdors(odorsDense) 
        nsensors = len(odors_testing[0]) 
        testingOdors = [] 
        for odor in odors_testing:
            testingOdors.append(copy.deepcopy(odor)) 

        # Set occlusion level
        if NOISE_LEVEL != 0:
            testingOdors = AddOcclusion(testingOdors, n=n_occlude, pList=noiseLevels)

        # Cover case of zero occlusion
        else:
            testingOdors = testingOdors

        wf = open("./pickle_files/multiOdorTest_noise" + experiment_name + ".pi", 'wb')
        pickle.dump(trainingOdors, wf, protocol=2) 
        pickle.dump(testingOdors, wf, protocol=2)
        wf.close()

        if VISUALISE_DATA:
            plot_data(odors_raw_training, odors_raw_testing, trainingOdors, testingOdors, odor_labels_training, odor_labels_testing, experiment_name)

        print("Done")