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



def run():
    pickle_files = './pickle_files'
    assert os.path.isdir(pickle_files)

    results_dir_parent = Path("./results/results_epl/")
    results_dir_parent.mkdir(exist_ok=True, parents=True)

    train_times = {}
    test_times = {}
    # Iterate over pickle_files with training/testing data for all experiments
    for file in os.listdir(pickle_files):

        if file[0]=='.':
            continue
        dst = pickle_files + "/" + file
        name = name = dst.split('/')[-1][:-3]
        #print(dst)

        # Make results dir
        results_dir = results_dir_parent.joinpath(file[:-3])
        print(results_dir)
        results_dir.mkdir(exist_ok=True, parents=True)

        # results_dir = Path("./results/results_epl/" + file[:-3] + "/")
        # #print(results_dir)
        # if not results_dir.exists():
        #     results_dir.mkdir(exist_ok=True, parents=True)
        # # if not os.path.exists(results_dir):
        # #     os.mkdir(results_dir)
        
        # name = dst.split('/')[-1].split('.')[0]

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
        start_time = time.time() 
        for i in range(0, len(trainingOdors)):
            print("Training odor " + str(i+1)) 
            sniff(trainingOdors[i], learn_flag=1)
            epl.GClayer.invokeNeurogenesis()
            sniff(trainingOdors[i], learn_flag=0)
        train_time = (time.time()-start_time)/len(trainingOdors)
        #Testing
        for i in range(0, len(testOdors)):
            sniff(testOdors[i], learn_flag=0)
            if(i%10==0 and i!=0):
                print(str(i) + " odors tested")
        test_time = (time.time()-start_time-train_time)/len(testOdors)
        train_times[name] = train_time
        test_times[name] = test_time

        # print("Simulation Duration = " + str(t3-t1) + "s")
        print("Training Duration = " + str(train_time) + "s")
        print("Testing Duration = " + str(test_time) + "s")

        #Readout
        sMatrix, odorClassification, netClassification = readout(epl.gammaCode, nOdors, nTestPerOdor)  

        # Save outputs
        # np.save(results_dir + "sMatrix.npy", np.array(sMatrix))
        # np.save(results_dir + "odorClassification.npy", np.array(odorClassification))
        # np.save(results_dir + "netClassification.npy", np.array(netClassification))
        # np.save(results_dir + "gammaCode.npy", np.array(epl.gammaCode))

        np.save(results_dir.joinpath("sMatrix.npy"), np.array(sMatrix))
        np.save(results_dir.joinpath("odorClassification.npy"), np.array(odorClassification))
        np.save(results_dir.joinpath("netClassification.npy"), np.array(netClassification))
        np.save(results_dir.joinpath("gammaCode.npy"), np.array(epl.gammaCode))

        #Plots
        # plots.plotFigure4a(epl.gammaCode, results_dir) 
        # plots.plotFigure4b(sMatrix, results_dir)
        # plots.plotFigure4d(epl.gammaCode, sMatrix, results_dir)

        print("done")
        # print(odorClassification)
        # print(netClassification)

        # exit()
    json.dump(train_times, open(results_dir_parent.joinpath("train_times.json"), "w"))
    json.dump(test_times, open(results_dir_parent.joinpath("test_times.json"), "w"))
    
if __name__ == '__main__':
    run()
