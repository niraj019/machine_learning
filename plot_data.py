import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
''''
Handle Data: Open the dataset from CSV and split into test/train datasets.
Similarity: Calculate the distance between two data instances.
Neighbors: Locate k most similar data instances.
Accuracy: Summarize the accuracy of predictions.
Main: Tie it all together.'''

number = int(input("number: "))

names = ['url_length', 'dots', 'url_numbers', 'dashes', 'class']

name = names[number]

''''Handle Data: Open the dataset from CSV and split into test/train datasets.
The first thing we need to do is load our data file. The data is in CSV format without a header line or any quotes.
 We can open the file with the open function and read the data lines using the reader function in the csv module.'''
df = pd.read_csv("malicious_vectors4.csv", header=1, names=names, error_bad_lines=False)
df.head()

print("Dataset Lenght:: ", len(df))
print("Dataset Shape:: ", df.shape)


X1 = np.array(df.ix[:,0])
X2 = np.array(df.ix[:,number])
Y = np.array(df['class'])
colors=[]
for y in Y:
	if y == 'bad':
		colors.append('red')
	else:
		colors.append('green')


plt.scatter(X1,X2,color=colors,marker='o', alpha=0.1)

plt.xlabel('URL lenght')
plt.ylabel(name)
plt.savefig(name+'.png')
plt.show()


