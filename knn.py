import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score

''''
Handle Data: Open the dataset from CSV and split into test/train datasets.
Similarity: Calculate the distance between two data instances.
Neighbors: Locate k most similar data instances.
Accuracy: Summarize the accuracy of predictions.
Main: Tie it all together.'''


names = ['url_length', 'dots', 'class']

''''Handle Data: Open the dataset from CSV and split into test/train datasets.
The first thing we need to do is load our data file. The data is in CSV format without a header line or any quotes.
 We can open the file with the open function and read the data lines using the reader function in the csv module.'''
df = pd.read_csv("malicious_vectors2.csv", header=1, names=names)
df.head()

X = np.array(df.ix[:, 0:2])
Y = np.array(df['class'])

'''Create four random training sets with an assigned size of 0.33'''
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


def predict(X_train, y_train, x_test, k):
    # this funcyion create two lists for distances and targets
    ''''args:
            X_train  = list
            y_train = list
            k = integer
        return:
            common_target = str
            '''
    distances = []
    targets = []
    ''''Similarity In order to make predictions we need to calculate the similarity between any two given data instances. 
    This is needed so that we can locate the k most similar data instances in the training dataset for a given member of 
    the test dataset and in turn make a prediction.: compute the euclidean distance'''
    for i in range(len(X_train)):
        distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
        # add it to list of distances
        distances.append([distance, i])

    # this line sort the distance list
    distances = sorted(distances)

    # this is for loop make a list of the k neighbors' targets
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])

    # return most common target
    common_target = Counter(targets).most_common(1)[0][0]
    return common_target

''''Neighbors: find the k-nearest Now that we have a similarity measure, we can use it collect the k most similar 
instances for a given unseen instance.'''

def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
    '''
    this function check if k larger than n
    args:
                X_train  = list
                y_train = list
                X_test = list
                predictions = list
                k = integer

            return:
                None
                '''
    if k > len(X_train):
        raise ValueError

    # predict for each testing observation
    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i, :], k))


predictions = []
try:
    ''''Accuracy: Summarize the accuracy of predictions.
'''
    kNearestNeighbor(X_train, y_train, X_test, predictions, 7)
    predictions = np.asarray(predictions)
    accuracy = accuracy_score(y_test, predictions) * 100
    print('\nAccuracy of the classifier is %d%%' % accuracy)

except ValueError:
    print("Error")
