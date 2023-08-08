Author: Nik Dennler, 2022, n.dennler2@herts.ac.uk

In this repo, we replicate and extend the analysis described in *Imam, N., Cleland, T.A. Rapid online learning and robust recall in a neuromorphic olfactory circuit. Nat Mach Intell 2, 181–191 (2020). https://doi.org/10.1038/s42256-020-0159-4*

Original code can be found here: https://github.com/ModelDBRepository/261864

To reproduce our analysis, follow those steps:
1. In Python 2.7, run `data/genData_extended.py`. This will load the data, preprocess if applicable, discretise, and produce pickle files with training and testing data. 
2. In Python 2.7, run `multiOdorTest_extended.py`. This will train the EPL network on the training set, compute the output when feeding in the testing set, then compute the similarity between the two. 
3. In Python 3+, run `produce_fig2.py`. This will produce Figure 2 of the manuscript, which displays a comparison of the computed similarities for the different settings.
