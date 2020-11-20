#Bryson Howell
#COSC 425 Team Project
#feature_analysis.py
#Does some analysis on the features in our dataset

import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize, scale, minmax_scale, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd

def main():

    features = ['num_files_changed','num_lines_changed','num_lines_per_file','gunning_fog_approx','num_chars','num_words','closes_issue', \
                'num_sentences', 'words_per_sentence', 'chars_per_word']
    targets = ['rank_message_length', 'rank_message_readability', 'rank_commit_size', 'rank_commit_overall']
    n_targets = len(targets)
    n_features = len(features)

    # Load in data as a csv
    csv = open("responses_2.csv")
    data = pd.read_csv('training_data_Hobson.csv', header = 0, skiprows=lambda x: x in [1], index_col=0)
    print(data.columns)

    csv.close()

    #Convert to to numpy
    X = np.array(data[features],dtype=np.float64())

    #Stdev test
    stdev_feat = np.zeros(n_features,dtype=np.float64())
    for i in range(0,n_features):
        stdev_feat[i] = np.std(X[:,i])

    plt.style.use('seaborn')
    fig = plt.figure()
    plt.title("Standard Deviation of Features")
    bars = np.arange(n_features)
    plt.bar(bars,stdev_feat)
    plt.ylabel("Standard Deviation")
    plt.xlabel("Feature")
    #plt.savefig("linearity_"+features[i],format='png')
    plt.show()

    #Standardize - MinMaxScaler
    X = MinMaxScaler().fit_transform(X)

#####################################################
#                                                   #
#               PCA STUFF                           #
#                                                   #
#####################################################

    #Reduce features with PCA
    c = n_features           #Set to select number of components
    pca = PCA(n_components=c)
    X = pca.fit_transform(X)

    #Make scree plot (set n_components=feat)
    fig = plt.figure()
    x = np.arange(n_features)+1
    plt.plot(x,np.cumsum(pca.explained_variance_ratio_))
    plt.plot(x,np.full(n_features,0.95))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Scree Plot')
    plt.grid()
    plt.show()

    print("number of components = %d" % pca.n_components_)
    print("Explained variance ratios: ")
    print(pca.explained_variance_ratio_)
    print("Variance explained: ")
    print(np.sum(pca.explained_variance_ratio_))

    comp = np.copy(abs(pca.components_))
    print()
    print("Principle components")
    print(comp)

    #Find weighted score of components
    wsums = np.zeros(n_features,np.float64())
    for i in range(0,c):
        for j in range(0,n_features):
            wsums[j] += (pca.explained_variance_ratio_[i]*comp[i][j])
    print("\nBest components (weighted sum)")
    print(features)
    print(wsums)


    return

if __name__ == '__main__':
    main()
