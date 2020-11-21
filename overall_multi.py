########### Created by Noah Hobson ###########

# Takes in data from GitHub commits and ranks
#   them based on their size and commit message
# This version uses a k-nearest neighbors
#   approach and has surprisingly good accuracy

###############################################

import warnings
import random
from numbers import Number
import numpy as np
import pandas as pd
import sys
from sklearn import svm, datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression as linreg

n_neighbors = 7
C=10

# By default, Sklearn forces warnings into your terminal.
# Here, we're writing a dummy function that overwrites the function
# that prints out numerical warnings.


def warn(*args, **kwargs):
    pass


warnings.warn = warn

# Load in data as a csv
csv = open("training_data_Hobson.csv")
data = pd.read_csv(csv)
csv.close()

# Two different feature sets for the two different models
size_feature_names = ["num_files_changed", "num_lines_changed"]
message_feature_names = ["gunning_fog_approx", "num_chars", "num_words",
                         "chars_per_word", "num_sentences", "words_per_sentence"]

size_features = np.array(data[size_feature_names], dtype=int)
message_features = np.array(data[message_feature_names], dtype=int)

size_scaler = MinMaxScaler().fit(size_features)
message_scaler = MinMaxScaler().fit(message_features)

scaled_size_features = size_scaler.transform(size_features)
scaled_message_features = message_scaler.transform(message_features)

# Grab our target vectors (commit size, message length and message readability)
size_output = np.array(data["rank_commit_size"], dtype=int)
length_output = np.array(data["rank_message_length"], dtype=int)
readability_output = np.array(data["rank_message_readability"], dtype=int)

# Start training the three different models (with multithreading support)
size_model = None
message_readability_model = None
message_length_model = None

if ("knn" in sys.argv):
    size_model = KNN(n_neighbors=n_neighbors, n_jobs=8)
    message_readability_model = KNN(n_neighbors=n_neighbors, n_jobs=8)
    message_length_model = KNN(n_neighbors=n_neighbors, n_jobs=8)
elif ("svc" in sys.argv or "svm" in sys.argv):
    size_model = SVC(C=C)
    message_readability_model = SVC(C=C)
    message_length_model = SVC(C=C)
else:
    size_model = linreg()
    message_readability_model = linreg()
    message_length_model = linreg()

size_model.fit(scaled_size_features, size_output)
message_readability_model.fit(scaled_message_features, readability_output)
message_length_model.fit(scaled_message_features, length_output)

predicted_length = message_length_model.predict(scaled_message_features)
predicted_readability = message_readability_model.predict(
    scaled_message_features)
predicted_size = size_model.predict(scaled_size_features)

# Test the models for accuracy
if ("knn" in sys.argv):
    print("Accuracy for message length with knn k=" + str(n_neighbors) +
        " neighbors: " + str(accuracy_score(length_output, predicted_length)))
    print("Accuracy for message readability with knn k=" + str(n_neighbors) +
      " neighbors: " + str(accuracy_score(readability_output, predicted_readability)))
    print("Accuracy for commit size with knn k=" + str(n_neighbors) +
      " neighbors: " + str(accuracy_score(size_output, predicted_size)))
elif ("svc" in sys.argv or "svm" in sys.argv):
    print("Accuracy for message length with svc C=" + str(C) + ": ", accuracy_score(length_output, predicted_length))
    print("Accuracy for message readability with svc C=" + str(C) + ": ", accuracy_score(readability_output, predicted_readability))
    print("Accuracy for commit size with svc C=" + str(C) + ": ", accuracy_score(size_output, predicted_size))
else:
    print("R² for message length: ", r2_score(length_output, predicted_length))
    print("R² for message readability: ", r2_score(readability_output, predicted_readability))
    print("R² for commit size: ", r2_score(size_output, predicted_size))



while True:
    try:
        input("\nPress Enter for more examples, or ctrl+c/ctrl+d to quit.")
        print("\n"*10)
        rand_num = random.randint(1, len(data) - 1)
        rand_size_input = [size_features[rand_num]]
        rand_scaled_size_input = [scaled_size_features[rand_num]]
        rand_message_input = [message_features[rand_num]]
        rand_scaled_message_input = [scaled_message_features[rand_num]]

        rank_size = size_model.predict(rand_scaled_size_input)
        rank_length = message_length_model.predict(rand_scaled_message_input)
        rank_readability = message_readability_model.predict(
            rand_scaled_message_input)

        print("\nCommit size: ",
              rand_size_input[0][0], " files, ", rand_size_input[0][1], " lines")
        print("Predicted commit rank: ", rank_size[0])
        print("True commit rank: ", size_output[rand_num])
        print("\n\tMessage: ", data["commit_message"][rand_num])
        print("\nPredicted message length rank: ", rank_length[0])
        print("True message length rank: ", length_output[rand_num])
        print("\nPredicted message readability rank: ", rank_readability[0])
        print("True message readability rank: ", readability_output[rand_num])
    except:
        print("\nExiting.\n")
        break
