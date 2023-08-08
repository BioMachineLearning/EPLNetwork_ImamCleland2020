# Original Authors:
# Imam, Nabil ; Cleland, Thomas [tac29 at cornell.edu];
# 2020
# https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=261864#tabs-1
# 
# Modified by Nik Dennler, 2023, n.dennler2@herts.ac.uk
# 
# ATTENTION: Run with Python 3!

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
new_rc_params = {"text.usetex": False, "svg.fonttype": "none"}
mpl.rcParams.update(new_rc_params)

from lib.plots import plot_similarity_comparison, plot_runtimes

if __name__ == '__main__':

    # 1. Plot similarity matrices
    files = [
        "multiOdorTest_noise0.6_090s_090s_SO_False_controltestFalse_samebinsTrue.pi",
        "multiOdorTest_noise0.6_015s_015s_SO_False_controltestFalse_samebinsTrue.pi",
        "multiOdorTest_noise0.6_090s_090s_SO_False_controltestTrue_samebinsTrue.pi",
        "multiOdorTest_noise0.6_090s_090s_SO_True_controltestTrue_samebinsTrue.pi",
        "multiOdorTest_noise0_090s_090s_SO_True_controltestTrue_samebinsTrue.pi",
    ]
    experiments = [
        "t=90s\n60% occlusion\nsame data for train / test\nno baseline subtraction",
        "t=15s (no odour)\n60% occlusion\nsame data for train / test\nno baseline subtraction",
        "t=90s\n60% occlusion\nsep. data for train / test\nno baseline subtraction",
        "t=90s\n60% occlusion\nsep. data for train / test\nbaseline subtraction",
        "t=90s\n0% occlusion\nsep. data for train / test\nbaseline subtraction",
        ]

    # EPL Network
    all_similarities = []
    for file in files:
        if file[0]=='.':
            continue
        results_dir = "./results_epl/" + file[:-3] + "/"
        print(results_dir)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        a = np.load(results_dir + "SMatrix.npy")
        all_similarities.append(a)
    plot_similarity_comparison(all_similarities, experiments, results_dir)

    # Clever Classifier
    all_similarities = []
    for file in files:
        if file[0]=='.':
            continue
        results_dir = "./results_hashtable/" + file[:-3] + "/"
        print(results_dir)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        a = np.load(results_dir + "SMatrix.npy")
        all_similarities.append(a)
    plot_similarity_comparison(all_similarities, experiments, results_dir) 

    # 2. Plot runtimes

    # Training runtimes
    n_train = 10
    t_train_epl_cpu = 109.945 / n_train         # Measured on a Macbook Pro (Apple M1 Pro)
    t_train_epl_loihi = 2.0*1e-3                # From paper
    t_train_clever_cpu = 6.914*1e-6 / n_train   # Measured on a Macbook Pro (Apple M1 Pro)

    # Testing runtimes
    n_test = 100
    t_test_epl_cpu = 876.901 / n_test           # Measured on a Macbook Pro (Apple M1 Pro)
    t_test_epl_loihi = 2.0*1e-3                 # From paper
    t_test_clever_cpu = 0.0956 / n_test         # Measured on a Macbook Pro (Apple M1 Pro)

    plot_runtimes(t_train_epl_cpu, t_train_epl_loihi, t_train_clever_cpu, t_test_epl_cpu, t_test_epl_loihi, t_test_clever_cpu, results_dir)