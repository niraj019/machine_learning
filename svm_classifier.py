#source: https://stackoverflow.com/questions/43284811/plot-svm-with-matplotlib

from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score

names = ['url_length', 'dots', 'url_numbers', 'dashes', 'class']

#data = pd.read_csv('./malicious_vectors3.csv', sep=',', header=None, error_bad_lines=False)
data = pd.read_csv('./malicious_vectors_4.csv', header=1, names=names, error_bad_lines=False)

print("Dataset Lenght:: ", len(data))
print("Dataset Shape:: ", data.shape)

print(data.head())


X = np.array(data.values[:, 0:4])
Y = np.array(data.values[:,4])


X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


clf = svm.SVC()
clf.fit(X_train, y_train)
print('fitted')
prediction = clf.predict(X_test)

#print(prediction)
print('accuracy: ', accuracy_score(y_test, prediction)*100)






