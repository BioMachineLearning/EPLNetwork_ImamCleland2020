# Original Authors:
# Imam, Nabil ; Cleland, Thomas [tac29 at cornell.edu];
# 2020
# https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=261864#tabs-1
# 
# Modified by Nik Dennler, 2023, n.dennler2@herts.ac.uk
# 
# ATTENTION: Run with Python 3!

import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
new_rc_params = {"text.usetex": False, "svg.fonttype": "none"}
mpl.rcParams.update(new_rc_params)

from lib.plots import plot_similarity_comparison, plot_runtimes

def run(results_dir_parent, figures_dir):    
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
        results_dir = results_dir_parent.joinpath("results_epl", file[:-3])
        # results_dir = Path("results/results_epl/" + file[:-3] + "/")
        # print(results_dir)
        results_dir.mkdir(exist_ok=True, parents=True)
        # if not os.path.exists(results_dir):
        #     os.mkdir(results_dir)
        a = np.load(results_dir.joinpath("sMatrix.npy"))
        # a = np.load(results_dir + "SMatrix.npy")
        all_similarities.append(a)
    plot_similarity_comparison(all_similarities, experiments, figures_dir, name="epl")

    # Hashtable Classifier
    all_similarities = []
    for file in files:
        if file[0]=='.':
            continue
        results_dir = results_dir_parent.joinpath("results_hashtable", file[:-3])
        # results_dir = Path("./results/results_hashtable/" + file[:-3] + "/")
        # print(results_dir)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        a = np.load(results_dir.joinpath("sMatrix.npy"))
        # a = np.load(results_dir + "SMatrix.npy")
        all_similarities.append(a)
    plot_similarity_comparison(all_similarities, experiments, figures_dir, name="hashtable")

    # 2. Plot runtimes

    # # Training runtimes
    # n_train = 10
    # t_train_epl_cpu = 34.939 / n_train          # Measured on a Macbook Pro (Apple M1 Pro)
    # t_train_epl_loihi = 2.0*1e-3                # From paper
    # t_train_clever_cpu = 6.914*1e-6 / n_train   # Measured on a Macbook Pro (Apple M1 Pro)

    # # Testing runtimes
    # n_test = 100
    # t_test_epl_cpu = 248.093 / n_test           # Measured on a Macbook Pro (Apple M1 Pro)
    # t_test_epl_loihi = 2.0*1e-3                 # From paper
    # t_test_clever_cpu = 0.0956 / n_test         # Measured on a Macbook Pro (Apple M1 Pro)

    # # Training runtimes
    # results_dir.joinpath("results_epl", "train_times.json")
    # t_train_epl_cpu = np.mean(list(json.load(open("./results/results_epl/train_times.json", "r")).values()))#["multiOdorTest_noise0.6_090s_090s_SO_False_controltestFalse_samebinsTrue"]
    t_train_epl_cpu = np.mean(list(json.load(open(results_dir_parent.joinpath("results_epl", "train_times.json"), "r")).values()))
    t_train_epl_loihi = 2.0*1e-3                # From paper
    # t_train_clever_cpu = np.mean(list(json.load(open("./results/results_hashtable/train_times.json", "r")).values()))#["multiOdorTest_noise0.6_090s_090s_SO_False_controltestFalse_samebinsTrue"]
    t_train_clever_cpu = np.mean(list(json.load(open(results_dir_parent.joinpath("results_hashtable", "train_times.json"), "r")).values()))

    # # Testing runtimes
    # t_test_epl_cpu = np.mean(list(json.load(open("./results/results_epl/test_times.json", "r")).values()))#["multiOdorTest_noise0.6_090s_090s_SO_False_controltestFalse_samebinsTrue"]
    t_test_epl_cpu = np.mean(list(json.load(open(results_dir_parent.joinpath("results_epl", "test_times.json"), "r")).values()))
    t_test_epl_loihi = 2.0*1e-3                 # From paper
    # t_test_clever_cpu = np.mean(list(json.load(open("./results/results_hashtable/test_times.json", "r")).values()))#["multiOdorTest_noise0.6_090s_090s_SO_False_controltestFalse_samebinsTrue"]
    t_test_clever_cpu = np.mean(list(json.load(open(results_dir_parent.joinpath("results_hashtable", "test_times.json"), "r")).values()))
    
    plot_runtimes(t_train_epl_cpu, t_train_epl_loihi, t_train_clever_cpu, t_test_epl_cpu, t_test_epl_loihi, t_test_clever_cpu, figures_dir)

if __name__ == '__main__':
    results_dir = Path("results_current/")
    figures_dir = Path("results/figures/")
    figures_dir.mkdir(exist_ok=True, parents=True)

    run(results_dir, figures_dir)