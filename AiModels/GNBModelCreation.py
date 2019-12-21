import csv
import sys
import random
from numpy import array
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pickle

filename = "RussianData_in.csv"

fields = []
matrix = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)

    for row in csvreader:
        matrix.append(row)


random.shuffle(matrix)

train_data = matrix[:int(0.8*len(matrix))]
test_data = matrix[int(0.8*len(matrix)):]

train_data = np.array(train_data, dtype=float)
test_data = np.array(test_data, dtype=float)


gnb = GaussianNB()
#Used for traits independent
#gnb = gnb.fit(train_data[:,2:], train_data[:,0])
#y_pred = gnb.predict(test_data[:,2:])

#Used for traits inclusive
gnb = gnb.fit(train_data[:,1:], train_data[:,0])
y_pred = gnb.predict(test_data[:,1:])

#print("Actual test values")
#print(test_data[:,0])
#print('\n')
#print("Predicted values")
#print(y_pred)
#print('\n')
differences = test_data[:,0] - y_pred
#print(differences)
a = differences.T

success = 0
for i in np.nditer(a):
    if i >= -30 and i <= 30:
        success += 1
print("%d successes of %d points" % (success, len(differences)))

pickle.dump(gnb, open( "GNBModel", "wb" ))

#print("Number of mislabeled points out of a total %d points : %d" % (test_data.shape[0],(test_data[:,0] != y_pred).sum()))
