import pickle
import numpy as np
import matplotlib.pyplot as plt
import random

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

labeldict = unpickle('../cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]


def cifar10_classifier_1nn(X = np.array([[]]), trdata = np.array([[]]), trlabel = np.array([[]]), limit = -1):
    if limit != -1:
        X = X[0:limit]

    labels = []

    for i in range (X.shape[0]):
        best = np.sum((X[i]-trdata[0])**2)
        bestLabel = trlabel[0]

        for j in range(trdata.shape[0]):
            dist = np.sum((X[i]-trdata[j])**2)
            if dist < best:
                best = dist
                bestLabel = trlabel[j]

        labels.append(bestLabel)

    return np.array(labels)

def class_acc(pred, gt, limit = -1):
    if limit != -1:
        gt = gt[0:limit]
    return np.sum(pred == gt) / pred.shape[0]


def cifar10_classifier_random(X):
    labels = [random.randrange(len(label_names)) for _ in range(X.shape[0])]

    return np.array(labels)


# read and store the test data
datadict = unpickle('./cifar-10-batches-py/test_batch')
unclassifiedX = datadict["data"]
classifyLabel = datadict["labels"]

unclassifiedX = unclassifiedX.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint32")
classifyLabel = np.array(classifyLabel)


# read and store the training data sets
X = np.array([])
Y = np.array([])
for i in range(1, 6):
    trainingdatadict = unpickle(f'../cifar-10-batches-py/data_batch_{i}')
    
    X = np.append(X,trainingdatadict["data"])
    Y = np.append(Y, trainingdatadict["labels"])

X = X.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint32")


# classifying using the random classifier
classification_rand = cifar10_classifier_random(X)
print(f" Success rate for random method classifier: {class_acc(classification_rand, Y) * 100:.2f}%")


# classifying using the 1-NN classifier
classification_1nn = cifar10_classifier_1nn(unclassifiedX, X, Y)
print(f" Success rate for 1nn method classifier: {class_acc(classification_1nn, classifyLabel) * 100:.2f}%")


def show():
    fig = plt.figure(figsize=(20, 20))
    columns = 4
    rows = 5

    # ax enables access to manipulate each of subplots
    ax = []

    for i in range(columns*rows):
        indexPic = random.randrange(0,10) 
        img = unclassifiedX[indexPic]
        # create subplot and append to ax
        ax.append( fig.add_subplot(rows, columns, i+1) )
        ax[-1].set_title(f"#{indexPic} : 1nn => {label_names[int(classification_1nn[indexPic])]}, gt => {label_names[int(classifyLabel[indexPic])]}")  # set title
        ax[-1].imshow(img)


    plt.show()

show()