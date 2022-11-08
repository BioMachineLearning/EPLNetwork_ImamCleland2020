# Original Authors:
# Imam, Nabil; Cleland, Thomas [tac29 at cornell.edu];
# 2020
# https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=261864#tabs-1
# 
# Modified by Nik Dennler, 2022, n.dennler2@herts.ac.uk
# 
# ATTENTION: Run with Python 2.7!

import time
import numpy as np
import re
import os

from lib.epl import EPL 
from lib.OSN import OSN_encoding
from lib.readout import readout
from lib import plots
import pickle

pickle_files = './pickle_files'

print(os.path.isdir(pickle_files))

# Iterate over pickle_files with training/testing data for all experiments
for file in os.listdir(pickle_files):
    if file[0]=='.':
        continue
    dst = pickle_files + "/" + file
    print(dst)
    results_dir = "./results/" + file[:-3] + "/"
    print(results_dir)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    name = dst.split('/')[-1].split('.')[0]
    print(name)

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
    t1 = time.time() 
    for i in range(0, len(trainingOdors)):
        print("Training odor " + str(i+1)) 
        sniff(trainingOdors[i], learn_flag=1)
        epl.GClayer.invokeNeurogenesis()
        sniff(trainingOdors[i], learn_flag=0)

    #Testing
    for i in range(0, len(testOdors)):
        sniff(testOdors[i], learn_flag=0)
        if(i%10==0 and i!=0):
            print(str(i) + " odors tested")
    t2 = time.time()
    print("Simulation Duration = " + str(t2-t1) + "s")

    #Readout
    sMatrix, odorClassification, netClassification = readout(epl.gammaCode, nOdors, nTestPerOdor)  

    # Save outputs
    np.save(results_dir + "sMatrix.npy", np.array(sMatrix))
    np.save(results_dir + "odorClassification.npy", np.array(odorClassification))
    np.save(results_dir + "netClassification.npy", np.array(netClassification))
    np.save(results_dir + "gammaCode.npy", np.array(epl.gammaCode))

    #Plots
    plots.plotFigure4a(epl.gammaCode, results_dir) 
    plots.plotFigure4b(sMatrix, results_dir)
    plots.plotFigure4d(epl.gammaCode, sMatrix, results_dir)

    print("done")
    print(odorClassification)
    print(netClassification)


