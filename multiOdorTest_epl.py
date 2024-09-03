# Original Authors:
# Imam, Nabil; Cleland, Thomas [tac29 at cornell.edu];
# 2020
# https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=261864#tabs-1
# 
# Modified by Nik Dennler, 2022, n.dennler2@herts.ac.uk
# 
# ATTENTION: Run with Python 3!

import time
import numpy as np
import re
import os
import json
from pathlib import Path

from lib.epl import EPL 
from lib.OSN import OSN_encoding
from lib.readout import readout
from lib import plots
import pickle



def run(dir_pickle_files, results_dir_parent):

    results_dir_parent.mkdir(exist_ok=True, parents=True)

    train_times = {}
    test_times = {}
    # Iterate over pickle_files with training/testing data for all experiments
    # for file in os.listdir(dir_pickle_files):
    for i, file in enumerate(dir_pickle_files.iterdir()):
        print(i, file.stem)

        if file.name[0]=='.':
            continue

        dst = file
        name = dst.name.split('/')[-1][:-3]

        # Make results dir
        results_dir = results_dir_parent.joinpath(file.stem)
        results_dir.mkdir(exist_ok=True, parents=True)

        # Load training and testing arrays
        rf = open(dst, "rb")
        trainingOdors = np.array(pickle.load(rf))
        testOdors = np.array(pickle.load(rf))
        rf.close()

        nOdors = len(trainingOdors) 
        nTestPerOdor = len(testOdors)/nOdors  
        print("Number of odors to train = " + str(len(trainingOdors))) 
        print("Number of odors to test = " + str(len(testOdors))) 

        #Network initialization
        nMCs = len(trainingOdors[0]) 
        GCsPerNeurogenesis = 5 
        nGCs = nMCs*GCsPerNeurogenesis*nOdors     #every MC has 5 GCs per odor  
        epl = EPL(nMCs, nGCs, GCsPerNeurogenesis) 

        #Sniff
        def sniff(odor, learn_flag=0, nGammaPerOdor=5, gPeriod=40):
            sensorInput = OSN_encoding(odor) 
            for j in range(0, nGammaPerOdor):
                for k in range(0, gPeriod): 
                    epl.update(sensorInput, learn_flag=learn_flag)
                    pass 
            epl.reset() 

        #Training
        t0 = time.time() 
        for i in range(0, len(trainingOdors)):
            # print("Training odor " + str(i+1)) 
            sniff(trainingOdors[i], learn_flag=1)
            epl.GClayer.invokeNeurogenesis()
            sniff(trainingOdors[i], learn_flag=0)
        t1 = time.time()
        #Testing
        for i in range(0, len(testOdors)):
            sniff(testOdors[i], learn_flag=0)
            # if(i%10==0 and i!=0):
            #     print(str(i) + " odors tested")
        t2 = time.time()
        train_time = t1-t0
        test_time = t2-t1
        train_times[name] = train_time/len(trainingOdors)
        test_times[name] = test_time/len(testOdors)

        print("Training Duration total = " + str(train_time) + "s")
        print("Testing Duration total = " + str(test_time) + "s")

        #Readout
        sMatrix, odorClassification, netClassification = readout(epl.gammaCode, nOdors, nTestPerOdor)  

        np.save(results_dir.joinpath("sMatrix.npy"), np.array(sMatrix))
        np.save(results_dir.joinpath("odorClassification.npy"), np.array(odorClassification))
        np.save(results_dir.joinpath("netClassification.npy"), np.array(netClassification))
        np.save(results_dir.joinpath("gammaCode.npy"), np.array(epl.gammaCode))

        #Plots
        # plots.plotFigure4a(epl.gammaCode, results_dir) 
        # plots.plotFigure4b(sMatrix, results_dir)
        # plots.plotFigure4d(epl.gammaCode, sMatrix, results_dir)

        # break
    
    json.dump(train_times, open(results_dir_parent.joinpath("train_times.json"), "w"))
    json.dump(test_times, open(results_dir_parent.joinpath("test_times.json"), "w"))
    
if __name__ == '__main__':
    dir_pickle_files = Path('pickle_files_current')
    dir_results = Path('results_current')
    run(dir_pickle_files, (dir_results / 'results_epl'))
