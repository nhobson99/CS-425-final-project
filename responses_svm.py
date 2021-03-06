import warnings
from numbers import Number
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

# By default, Sklearn forces warnings into your terminal.
# Here, we're writing a dummy function that overwrites the function
# that prints out numerical warnings.
# (You probably don't want to do this in any of your projects!)


def warn(*args, **kwargs):
    pass


warnings.warn = warn

# Load in data as a csv
csv = open("responses_2.csv")
data = pd.read_csv(csv)
csv.close()

features = ["num_files", "num_lines"]

# Separate our features from our targets.
X = np.array(data[features], dtype=int)
X[:, 0] = X[:, 0].clip(1, 10)
X[:, 1] = X[:, 1].clip(1, 160)

Y = np.array(data["rank_commit_size"])

# X = StandardScaler().fit_transform(X)

# Use Sklearn to get splits in our data for training and testing.
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=0)
x_train, x_test, y_train, y_test = X, X, Y, Y
y_train_converted = y_train

# Now, we create several instances of SVCs that utilize varying kernels.
# We're not normalizing our data here because we want to plot the support vectors.
svc = svm.SVC(kernel='linear', C=10).fit(x_train, y_train)

# Now, we run our test data through our trained models.
predicted_linear = svc.predict(x_test)

# Print our accuracies.
print("SVC + Linear\t\t-> " + str(accuracy_score(y_test, predicted_linear)))


# # # # # # # # # # # # #
# PLOTTING CODE STARTS  #
# # # # # # # # # # # # #

# create a mesh to plot in
h = 0.1  # step size in the mesh
xaxis = x_test[:, 0]
yaxis = x_test[:, 1]
x_min, x_max = xaxis.min()-1, xaxis.max()+1
y_min, y_max = yaxis.min()-1, yaxis.max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# If we wanted to set a color scheme for our plot, we could do so here.
# For example:

clf = svc

plt.set_cmap(plt.cm.RdYlGn)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Put the result into a color plot
# Apply
plt.contourf(xx, yy, Z)
plt.axis('tight')

# Plot also the training points
y_pred = clf.predict(x_test)
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, edgecolor='black')
plt.title("Linear Kernel SVM")

plt.axis('tight')
plt.show()

# # # # # # # # # # # # #
# PLOTTING CODE Ends  #
# # # # # # # # # # # # #
