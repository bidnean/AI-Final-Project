import csv
import random
from numpy import array
#import sys
from sklearn.naive_bayes import GaussianNB

filename = "Russian Data.csv"

fields = []
matrix = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)

    for row in csvreader:
        matrix.append(row)
    print("Total number of rows: %d"%(csvreader.line_num))
    #print(matrix)

#for i in range(len(matrix)):
    #for j in range(len(matrix[i])):
        #if not matrix[i][j]:
            #matrix[i][j] = '0'


#CODE TO CREATE AN AI AI MODEL
random.shuffle(matrix)
#print(matrix)
print(len(matrix))

train_data = matrix[:int(0.8*len(matrix))]
train_data = array(train_data)
print(train_data)
print(type(train_data))


test_data = matrix[int(0.8*len(matrix)):]
test_data = array(test_data)
print(test_data)
print(type(test_data))

print(fields[3])

#THIS PART IS FROM THE ORIGINAL LAB, AND I CAN'T FIGURE OUT HOW TO SLICE IT CORRECTLY
#It should be fit(train_data, train_target) then predict(test_data)
gnb = GaussianNB()
#train_data[:,:3] goes from row 0 to the end and then column 0 to 3. I can't figure out how to skip over columns
gnb = gnb.fit(train_data[:,:3], train_data[:,3])
y_pred = gnb.predict(test_data[:,1:])
