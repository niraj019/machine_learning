#source: http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
#source: http://scikit-learn.org/stable/modules/tree.html

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz 




names = ['url_length', 'dots', 'url_numbers', 'dashes', 'class']

#data = pd.read_csv('./malicious_vectors3.csv', sep=',', header=None, error_bad_lines=False)
data = pd.read_csv('./malicious_vectors4.csv', header=1, names=names, error_bad_lines=False)

print("Dataset Lenght:: ", len(data))
print("Dataset Shape:: ", data.shape)

print(data.head())


X = np.array(data.values[:, 0:4])
Y = np.array(data.values[:,4])


	
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)



clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=10, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)


clf = clf_entropy.fit(X_train, y_train)

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=['url_length', 'dots', 'url_numbers', 'dashes'],  
                         class_names=['good','bad'],  
                         filled=True, rounded=True,  
                         special_characters=True)  


#dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("tree")
graph

y_pred_en = clf_entropy.predict(X_test)


	
print("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)







