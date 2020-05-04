from __future__ import print_function, division
import math
from sklearn import datasets
from numpy import genfromtxt
import numpy

def tanH(x):
    return 2 / (1 + numpy.exp(-2*x)) - 1

def gradientTanH(x):
    return 1 - numpy.power(tanH(x), 2)

def CrossEntropy(y, p):
    # to avoid division by zero
    p = numpy.clip(p, 1e-15, 1 - 1e-15)
    return - y * numpy.log(p) - (1 - y) * numpy.log(1 - p)

def gradientCrossEntropy(y, p):
    # to avoid division by zero
    p = numpy.clip(p, 1e-15, 1 - 1e-15)
    return - (y / p) + (1 - y) / (1 - p)


dataTrain = numpy.loadtxt("MACHINE_LEARNING/Neural_Network_fromScratch/train_set_d_minus_4")
n_hidden=5
learning_rate=0.01

X=dataTrain[:, [0, 1]]
y=dataTrain[:, [2]]
print(X.shape)
print(X)

dataVal = numpy.loadtxt("MACHINE_LEARNING/Neural_Network_fromScratch/validation_set_d_minus_4")

Xval=dataVal[:, [0, 1]]
yval=dataVal[:, [2]]

dataTest = numpy.loadtxt("MACHINE_LEARNING/Neural_Network_fromScratch/test_set_2_d_1")

Xtest=dataTest[:, [0, 1]]
ytest=dataTest[:, [2]]

n_samples, n_features = X.shape
_, n_outputs = y.shape

#hiddenLayer
limit   = math.sqrt(3/n_features)
W  = numpy.random.uniform(-limit, limit, (n_features, n_hidden))
w0 = numpy.zeros((1, n_hidden))

#outputLayer
limit   = math.sqrt(3/n_hidden)
V  = numpy.random.uniform(-limit, limit, (n_hidden, n_outputs))
v0 = numpy.zeros((1, n_outputs))

max=0
for i in range(25):
        
    a=(i%1000*1)

    #FORWARDPROB
    # HIDDEN LAYER
    hidden_input = X[a:a+1].dot(W) + w0
    hidden_output = tanH(hidden_input)
    # OUTPUT LAYER
    output_layer_input = hidden_output.dot(V) + v0
    y_pred = tanH(output_layer_input)


    #BACKPROB
    # OUTPUT LAYER
    grad_wrt_out_l_input = gradientCrossEntropy(y[a:a+1], y_pred) * gradientTanH(output_layer_input)
    grad_v = hidden_output.T.dot(grad_wrt_out_l_input)
    grad_v0 = numpy.sum(grad_wrt_out_l_input, axis=0, keepdims=True)
    # HIDDEN LAYER
    grad_wrt_hidden_l_input = grad_wrt_out_l_input.dot(V.T) * gradientTanH(hidden_input)
    grad_w = X[a:a+1].T.dot(grad_wrt_hidden_l_input)
    grad_w0 = numpy.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)

    V  -= learning_rate * grad_v
    v0 -= learning_rate * grad_v0
    W  -= learning_rate * grad_w
    w0 -= learning_rate * grad_w0

    print("Learned hidden layer weights after the iteration:")
    print(W)
    print("Learned output layer weights after the iteration:")
    print(V)
    print("-------")
    hidden_input = Xval.dot(W) + w0
    hidden_output = tanH(hidden_input)
    output_layer_input = hidden_output.dot(V) + v0
    y_pred = tanH(output_layer_input)
    
    correct=0

    for j in range(1000):

        if(y_pred[j]>0 and yval[j]==1):
            correct+=1
        if(y_pred[j]<0 and yval[j]==-1):
            correct+=1

    print(correct)
    if(correct>max):
        max=correct
    if(correct==1000):
        print(i)
        break


print("-----------------------------------------")
print("Lets test our weights:")


print(X.shape)
hidden_input = Xtest.dot(W) + w0
hidden_output = tanH(hidden_input)
output_layer_input = hidden_output.dot(V) + v0
y_pred = tanH(output_layer_input)

correct=0
for j in range(2000):

    if(y_pred[j]>0 and ytest[j]==1):
        correct+=1
    if(y_pred[j]<0 and ytest[j]==-1):
        correct+=1
print("Accuracy on test set is:")
print((correct/2000)*100)
