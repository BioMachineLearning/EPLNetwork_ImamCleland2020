Author: Nik Dennler, 2023, n.dennler2@herts.ac.uk

In this repo, we replicate and extend the analysis described in *Imam, N., Cleland, T.A. Rapid online learning and robust recall in a neuromorphic olfactory circuit. Nat Mach Intell 2, 181–191 (2020). https://doi.org/10.1038/s42256-020-0159-4*

Original code can be found here: https://github.com/ModelDBRepository/261864

To reproduce the full analysis, follow those steps:
1. Run `genData_extended.py`. This will load the data, preprocess if applicable, discretise, and produce pickle files with training and testing data. 
2. Run `multiOdorTest_epl.py`. For all configurations, this will train the EPL network on the training set, then compute the restored signals as the networks response to the testing set, and finally compute the similarity between the restored and trained.
3. Run `multiOdorTest_hashtable.py`. For all configurations, this will populate the hash table with data from the the training set, then restore the signal from the testing set based on maximal overlap, and finally compute the similarity between restored and trained.
4. Run `produce_figures.py`. This will produce Figures 1, 2 and S2 of the manuscript, displays a comparison of the computed similarities for the different settings as well as of the different runtimes.
