import tensorflow as tf
from tensorflow import keras
import numpy as np

import pickle

# Reading the data to container
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

labeldict = unpickle('./cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

datadict = unpickle('./cifar-10-batches-py/test_batch')
unclassifiedX = datadict["data"]
classifyLabel = datadict["labels"]

unclassifiedXc = unclassifiedX.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint32")
unclassifiedX = unclassifiedX.reshape(10000, 3072).astype("uint32")
classifyLabel = np.array(classifyLabel)

X = np.array([])
Y = np.array([])
for i in range(1, 6):
    trainingdatadict = unpickle(f'./cifar-10-batches-py/data_batch_{i}')
    
    X = np.append(X, trainingdatadict["data"])
    Y = np.append(Y, trainingdatadict["labels"])

Xc = X.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint32") 
X = X.reshape(50000, 3072).astype("uint32")

def class_acc(pred, gt):
    return np.sum(pred == gt) / pred.shape[0]

# Onehot
def onehot(x):
    res = np.array([])

    for a in x:
        temp = [0 for _ in range(10)]
        temp[int(a)] = 1
        res = np.append(res, np.array(temp))

    return res.reshape(x.shape[0], 10).astype("int")


# Normalize the input
X = X.astype("float32") / 255
Xc = Xc.astype("float32") / 255

unclassifiedX = unclassifiedX.astype("float32") / 255
unclassifiedXc = unclassifiedXc.astype("float32") / 255

def cifar10_neural_full_connect(X, Y, unclassifiedX):
    Y = onehot(Y)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(50, input_dim=3072, activation="sigmoid"),
            tf.keras.layers.Dense(50, activation="sigmoid"),
            tf.keras.layers.Dense(10, activation="sigmoid")
        ]
    )

    model.summary()
    model.compile(
        loss = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO),
        optimizer = keras.optimizers.SGD(learning_rate = 0.25),
        metrics =["mse"],
        )
        
    model.fit(X, Y, epochs = 50,verbose = 1)

    predicts = model.predict(unclassifiedX)
    labels = np.array([int(np.argmax(predicts[i])) for i in range(len(predicts))])

    return labels

def cifar10_neural_convolution(X, Y, unclassifiedX):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape= (32,32,3)),
            tf.keras.layers.Conv2D(32, 3, padding = "valid", activation = "relu"),
            tf.keras.layers.MaxPooling2D(pool_size = (2,2)),            
            tf.keras.layers.Conv2D(64, 3, activation = "relu"),
            tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
            tf.keras.layers.Conv2D(128, 3, activation = "relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation = "relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    model.summary()
    model.compile(
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        optimizer = keras.optimizers.Adam(learning_rate = 3e-4),
        metrics =["accuracy"])
        
    model.fit(X, Y, epochs = 10,verbose = 2)

    predicts = model.predict(unclassifiedX)
    labels = np.array([int(np.argmax(predicts[i])) for i in range(len(predicts))])

    return labels


# Train and run the models
neural_full = cifar10_neural_full_connect(X, Y, unclassifiedX)
neural_convolutional = cifar10_neural_convolution(Xc, Y, unclassifiedXc)


# Print out the accuracy of the models
print()
print("================================================================================================================")
print()

print(f'The accuracy of the fully connected neural network is: {class_acc(neural_full, classifyLabel)*100:.2f}%')
print()
print(f'The accuracy of the convolutional neural network is: {class_acc(neural_convolutional, classifyLabel)*100:.2f}%')