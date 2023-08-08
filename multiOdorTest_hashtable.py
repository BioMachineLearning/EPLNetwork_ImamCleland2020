import os
import time
import pickle
import numpy as np
from lib import plots

class HashTableClassifier():
    # def __init__(self) -> None:
    #     pass
        
    def train(self, training_data):        
        self.learned_odour_representations = {idx: odour for idx, odour in enumerate(training_data)}

    def denoise_test(self, testing_data):
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
        sMatrix = np.zeros((len(self.denoised_testing_data)*5, len(self.learned_odour_representations)))
        for idx_test, testing_odour in self.denoised_testing_data.items():
            for idx_train, training_odour in self.learned_odour_representations.items():
                similarity = np.count_nonzero(training_odour==testing_odour) / 72
                for i in range(5):
                    sMatrix[5*idx_test+i, idx_train] = similarity
        return sMatrix


if __name__ == '__main__':
    # pickle_files = './pickle_files_cleverclassifier'
    pickle_files = './pickle_files'

    print(os.path.isdir(pickle_files))

    # Iterate over pickle_files with training/testing data for all experiments
    for file in os.listdir(pickle_files):
        if file[0]=='.':
            continue
        dst = pickle_files + "/" + file
        print(dst)
        results_dir = "./results_hashtable/" + file[:-3] + "/"
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

        hashtable_classifier = HashTableClassifier()
        start_time = time.time()
        hashtable_classifier.train(trainingOdors)
        train_time = time.time()-start_time
        denoised_test = hashtable_classifier.denoise_test(testOdors)
        test_time = time.time()-start_time-train_time
        print(train_time, test_time)#/len(testOdors))
        sMatrix = hashtable_classifier.get_similarity_matrix()

        # Save outputs
        np.save(results_dir + "sMatrix.npy", np.array(sMatrix))
        plots.plotFigure4b(sMatrix, results_dir)
        print("done")



