import os
import time
import pickle
from pathlib import Path
import json

import numpy as np
from lib import plots

class HashTableClassifier():

    def train(self, training_data):
        """
        The train method assigns learned odour representations to the given training data.

        Parameters:
        - training_data (list): A list of training data odours.

        Returns:
        None
        """     
        self.learned_odour_representations = {idx: odour for idx, odour in enumerate(training_data)}

    def denoise_test(self, testing_data):
        """
        The denoise_test method compares each testing odour with the learned odour representations to find the best matching odour.
        It then assigns the best matching learned odour representation to the denoised_testing_data dictionary.

        Parameters:
        - testing_data (list): A list of testing data odours.

        Returns:
        None
        """
        self.denoised_testing_data = {}
        for idx_test, testing_odour in enumerate(testing_data):
            best_matching, idx_best_matching = 0, None
            for idx_train, training_odour in self.learned_odour_representations.items():
                # if best_matching < np.count_nonzero(training_odour==testing_odour):
                #     best_matching, idx_best_matching = np.count_nonzero(training_odour==testing_odour), idx_train
                if best_matching < sum(training_odour==testing_odour):
                    best_matching, idx_best_matching = sum(training_odour==testing_odour), idx_train

            self.denoised_testing_data[idx_test] = self.learned_odour_representations[idx_best_matching]
    
    def get_similarity_matrix(self):
        """
        The get_similarity_matrix method calculates the similarity between each testing odour and each learned odour representation.
        It returns a 2D numpy array with the similarity values.

        Parameters:
        None

        Returns:
        sMatrix (numpy.ndarray): A 2D numpy array with the similarity values between testing and learned odours.
        """
        sMatrix = np.zeros((len(self.denoised_testing_data)*5, len(self.learned_odour_representations)))
        for idx_test, testing_odour in self.denoised_testing_data.items():
            for idx_train, training_odour in self.learned_odour_representations.items():
                similarity = np.count_nonzero(training_odour==testing_odour) / 72
                for i in range(5):
                    sMatrix[5*idx_test+i, idx_train] = similarity
        return sMatrix

# def run():
def run(dir_pickle_files, results_dir_parent):

    results_dir_parent.mkdir(exist_ok=True, parents=True)
    assert os.path.isdir(dir_pickle_files)

    # Iterate over pickle_files with training/testing data for all experiments
    train_times = {}
    test_times = {}
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

        # Initiate classifier
        hashtable_classifier = HashTableClassifier()

        # Run training
        t0 = time.time()
        hashtable_classifier.train(trainingOdors)
        t1 = time.time()

        # Run testing
        denoised_test = hashtable_classifier.denoise_test(testOdors)
        t2 = time.time()
        train_time = t1-t0
        test_time = t2-t1

        train_times[name] = train_time/len(trainingOdors)
        test_times[name] = test_time/len(testOdors)

        print("Training Duration total = " + str(train_time) + "s")
        print("Testing Duration total = " + str(test_time) + "s")

        # Extract similarity matrix
        sMatrix = hashtable_classifier.get_similarity_matrix()

        # Save outputs
        np.save(results_dir.joinpath("sMatrix.npy"), np.array(sMatrix))

        #plots.plotFigure4b(sMatrix, results_dir)
        # print("done")
    
    json.dump(train_times, open(results_dir_parent.joinpath("train_times.json"), "w"))
    json.dump(test_times, open(results_dir_parent.joinpath("test_times.json"), "w"))

if __name__ == '__main__':
    dir_pickle_files = Path('pickle_files_current')
    dir_results = Path('results_current')
    run(dir_pickle_files, (dir_results / 'results_hashtable'))