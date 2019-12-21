import csv
import sys
import random
from numpy import array
import numpy as np
from sklearn.linear_model import LinearRegression
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


linr = LinearRegression()
#Used for when traits are independent
#linr = linr.fit(train_data[:,2:], train_data[:,0])
#y_pred = linr.predict(test_data[:,2:])

#Used for traits inclusive
linr = linr.fit(train_data[:,1:], train_data[:,0])
y_pred = linr.predict(test_data[:,1:])

#print("Actual test values")
#print(test_data[:,0])
#print('\n')
#print("Predicted values")
#print(y_pred)
#print('\n')
differences = test_data[:,0] - y_pred.astype(int)

choppedList = differences % 10
finalList = differences - choppedList

a = finalList.T

success = 0
for i in np.nditer(a):
    if i >= -30 and i <= 30:
        success += 1
print("%d successes of %d points" % (success, len(finalList)))

pickle.dump(linr, open( "LinRModel", "wb" ))

#print("Number of mislabeled points out of a total %d points : %d" % (test_data.shape[0],(test_data[:,0] != y_pred).sum()))
