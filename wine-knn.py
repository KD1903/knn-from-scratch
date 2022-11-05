import csv
from math import sqrt

with open('wine_train.txt', 'r') as csvfile:
    lines = csv.reader(csvfile)
    data = list(lines)

    traindata = []
    trainlabel = []

    #split data and label
    for i in data:
        traindata.append(list(map(float, i[:-1])))
        trainlabel.append(int(i[-1]))


with open('wine_test.txt', 'r') as csvfile:
    lines = csv.reader(csvfile)
    data = list(lines)

    testdata = []
    testlabel = []

    #split data and label
    for i in data:
        testdata.append(list(map(float, i[:-1])))
        testlabel.append(int(i[-1]))

def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

def myKNN(traindata, trainlabel, testdata, k):
    testpredict = []

    #iterate through all testdata
    for testrow in testdata:
        distances = list()

        #find euclidean distance from all train data 
        for i in range(len(traindata)):
            trainrow = traindata[i]
            trainrowlabel = trainlabel[i]

            dist = euclidean_distance(testrow, trainrow)
            distances.append((trainrow, trainrowlabel, dist))

        #sort by distance score
        distances.sort(key=lambda tup: tup[2])

        predictions = list()
        
        #take top k values of predicted data
        for i in range(k):
            predictions.append(distances[i][1])

        #find final prediction from top k predictions
        output_values = predictions
        prediction = max(set(output_values), key=output_values.count)
        
        #save prediction
        testpredict.append(prediction)

    return testpredict

k = 25
testpredict = myKNN(traindata, trainlabel, testdata, k)

print(f"k = {k}")
print("predicted and actual class")

correct_predictions = 0

for i in range(len(testpredict)):
    print(testpredict[i], "\t\t", testlabel[i])

    if testpredict[i] == testlabel[i]:
        correct_predictions += 1

print("correct", correct_predictions, "total", len(testpredict), "percentage", correct_predictions*100/len(testpredict))