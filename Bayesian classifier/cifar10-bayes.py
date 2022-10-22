import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import skimage.transform as skt

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

labeldict = unpickle('./cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

datadict = unpickle('./cifar-10-batches-py/test_batch')
unclassifiedX = datadict["data"]
classifyLabel = datadict["labels"]

unclassifiedX = unclassifiedX.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint32")
classifyLabel = np.array(classifyLabel)

# Reading the training data
X = np.array([])
Y = np.array([])
for i in range(1, 6):
    trainingdatadict = unpickle(f'./cifar-10-batches-py/data_batch_{i}')
    
    X = np.append(X,trainingdatadict["data"])
    Y = np.append(Y, trainingdatadict["labels"])

X = X.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint32")


# cifar10_color function 
def cifar10_color(X, size=1):
    if size > 1:
        return np.array([skt.resize(data, output_shape=(size,size), preserve_range=True) for data in X]).reshape(X.shape[0], size * size, 3)
    return np.array([skt.resize(data, output_shape=(size,size), preserve_range=True) for data in X]).reshape(X.shape[0], 3)


# The naive method of Bayers classification
def cifar_10_naivebayers_learn(Xp, Y):
    mean = np.array([])
    sig = np.array([])
    p = np.array([])

    for i in range(10):
        mean = np.append(mean, np.mean(Xp[Y==i].T, axis=1))
        sig = np.append(sig, np.array([np.var(Xp[Y==i][:,j]) ** (1/2) for j in range(3)]))
        p = np.append(p, len(Xp[Y==i]) / len(Xp))

    return mean.reshape(10, 3), sig.reshape(10, 3), p

def fast_posterior_prob(x, classIndex, mu, sigma, p):
    numerator = 1
    for i in range(3):
        numerator *= stats.norm.pdf(x[i], mu[classIndex][i], sigma[classIndex][i])

    return numerator

def cifar10_classifier_naivebayes(x, mu, sigma, p):
    bestLabel = []

    for i in range(len(x)):
        bestYet = fast_posterior_prob(x[i], 0, mu, sigma, p)
        bestLabelYet = 0
        for classIndex in range(1, 10):
            prob = fast_posterior_prob(x[i], classIndex, mu, sigma, p)
            if prob > bestYet:
                bestYet = prob
                bestLabelYet = classIndex

        bestLabel.append(bestLabelYet)

    return np.array(bestLabel)


def class_acc(pred, gt):
    return np.sum(pred == gt) / pred.shape[0]


# The Bayers classification
def cifar_10_bayes_learn(Xf, Y, size=1):
    mean = np.array([])
    sig = np.array([])
    p = np.array([])

    for i in range(10):
        mean = np.append(mean, np.mean(Xf[Y==i].T.reshape(3*size*size,-1), axis=1))
        sig = np.append(sig, np.cov(Xf[Y==i].T.reshape(3*size*size,-1)))
        p = np.append(p, len(Xf[Y==i]) / len(Xf))

    return mean.reshape(10, size*size*3), sig.reshape(10, size*size*3, size*size*3), p

def cifar10_classifier_bayes(X, mu, sigma, p, size = 1):
    x = np.array([])
    for i in range(X.shape[0]):
        x = np.append(x, X[i].T.flatten())
    x = x.reshape(10000, 3*size*size)

    prob = np.array([])
    for classIndex in range(0, 10):
        prob = np.append(prob, stats.multivariate_normal.logpdf(x, mean=mu[classIndex], cov=sigma[classIndex]))

    prob = prob.reshape(10, 10000).T
    return np.array([int(np.argmax(prob[i])) for i in range(len(prob))])



# Testing the classification methods

# Using the naive Bayers classification approach
Xp = cifar10_color(X)
mean, sigma, p = cifar_10_naivebayers_learn(Xp, Y)

unclassifiedXp = cifar10_color(unclassifiedX)
naivebayes = cifar10_classifier_naivebayes(unclassifiedXp, mean, sigma, p)
print(f"The accuracy of the naive Bayers classification: {class_acc(naivebayes, classifyLabel)*100}%")
print()

# Using the Bayers classification approach
mean, sig, p = cifar_10_bayes_learn(Xp, Y)
bayes = cifar10_classifier_bayes(unclassifiedXp, mean, sig, p)
print(f"The accuracy of the Bayers classification: {class_acc(bayes, classifyLabel) * 100}%")
print()