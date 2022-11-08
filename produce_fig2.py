# Original Authors:
# Imam, Nabil ; Cleland, Thomas [tac29 at cornell.edu];
# 2020
# https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=261864#tabs-1
# 
# Modified by Nik Dennler, 2022, n.dennler2@herts.ac.uk
# 
# ATTENTION: Run with Python 3!

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
new_rc_params = {"text.usetex": False, "svg.fonttype": "none"}
mpl.rcParams.update(new_rc_params)

def plot_similarity_comparison(sMatrix_all, experiments, results_dir): 
    fig, ax = plt.subplots(figsize=(18/1.4, 5/1.4))
    xticks = []
    xtick_labels = []
    for i, (sMatrix, experiment) in enumerate(zip(sMatrix_all, experiments)):
        n_test = 10
        gamma = 5; 
        testOdorID = 0;              #sniff ID of test odor 
        test1_gamma = sMatrix[gamma-1:(testOdorID+1)*(gamma)*n_test:gamma]
        bars = np.median(test1_gamma,axis=0)
        std = np.std(test1_gamma, axis=0)
        q25 = np.median(test1_gamma, axis=0) - np.quantile(test1_gamma, axis=0, q=0.25)
        q75 = - np.median(test1_gamma, axis=0) + np.quantile(test1_gamma, axis=0, q=0.75)
        w = 0.15        #width of bars
  
        xticks.append(i)
        xtick_labels.append(experiment)
        
        fig = plt.subplot(111)
        opacity = 0.5; 
        
        bar1 = ax.bar([i-2*w], bars[0], width = w, color = 'blue', alpha = opacity, align='center')
        bar2 = ax.bar([i-w], bars[4], width = w, color = 'violet', alpha = opacity, align='center')
        bar3 = ax.bar([i], bars[5], width = w, color = 'red', alpha = opacity, align='center')
        bar4 = ax.bar([i+w], bars[1], width = w, color = 'orange', alpha=opacity, align='center')
        bar5 = ax.bar([i+2*w], bars[2], width = w, color = 'green', alpha=opacity, align='center')

        errbar1 = ax.errorbar(np.expand_dims(i-2*w, axis=-1), np.expand_dims(bars[0], axis=-1), yerr=np.expand_dims([q25[0], q75[0]], axis=-1), color = 'darkslategray') 
        errbar2 = ax.errorbar(np.expand_dims(i-w, axis=-1), np.expand_dims(bars[4], axis=-1), yerr=np.expand_dims([q25[4], q75[4]], axis=-1), color = 'darkslategray')
        errbar3 = ax.errorbar(np.expand_dims(i, axis=-1), np.expand_dims(bars[5], axis=-1), yerr=np.expand_dims([q25[5], q75[5]], axis=-1), color = 'darkslategray')
        errbar4 = ax.errorbar(np.expand_dims(i+w, axis=-1), np.expand_dims(bars[1], axis=-1), yerr=np.expand_dims([q25[1], q75[1]], axis=-1), color = 'darkslategray')
        errbar5 = ax.errorbar(np.expand_dims(i+2*w, axis=-1), np.expand_dims(bars[2], axis=-1), yerr=np.expand_dims([q25[2], q75[2]], axis=-1), color = 'darkslategray')

        ax.axvline(0.5, linestyle='--', c='k')
        ax.set_ylim(0, 1.02)

    fig.legend( (bar1, bar2, bar3, bar4, bar5), ('Toluene', 'Ammonia', 'Acetone', 'Benzene', 'Methane'), loc = 'upper right')
    ax.set_ylabel('Jaccard Similarity Coefficient')
    ax.set_xticks(xticks, xtick_labels)
    plt.grid(axis='y')
    plt.savefig(results_dir +  "fig_comparison_new_median.png", dpi=300, bbox_inches='tight')
    plt.savefig(results_dir + "fig_comparison_new_median.svg", bbox_inches='tight')
    plt.show()
    plt.close()

if __name__ == '__main__':
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

    all_similarities = []
    for file in files:
        if file[0]=='.':
            continue
        results_dir = "./results/" + file[:-3] + "/"
        print(results_dir)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        a = np.load(results_dir + "SMatrix.npy")
        all_similarities.append(a)

    plot_similarity_comparison(all_similarities, experiments, results_dir); 

    print(results_dir)
    print("Done")

