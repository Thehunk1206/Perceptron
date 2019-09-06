import numpy as np
import time


def sigmoid(x,derivative=False):
    if derivative==True:
        return x*(1-x)
    return 1/(1+ np.exp(-x))


training_inputs = np.array([[1,0,1],
                            [1,1,1],
                            [1,0,1],
                            [1,0,1],
                            [0,1,0],
                            [1,1,1],
                            [1,0,0]])

training_outputs = np.array([[0,1,0,0,1,1,1]]).T


#initializing random weights
np.random.seed(0)
weights = 2*np.random.random((3,1))
print('Initializing weights')
print(weights)
print("Starting traing process\n")
time.sleep(1)


#training
for i in range(20000):

    input_layer = training_inputs

    output = sigmoid(np.dot(input_layer,weights))

    error = training_outputs - output

    if i%100 == 0:
        print(error.flatten())
        time.sleep(0.04)

    adjustments = error * sigmoid(output,derivative=True)

    weights+=np.dot(input_layer.T,adjustments)
time.sleep(2)
print("===========================================")
print("weights after training")
time.sleep(2)
print(weights,"\n")
time.sleep(2)
print("===========================================")
print("output after training\n")

print("Actual output",training_outputs.flatten(),"\n")
print("Output by perceptron: ",output.flatten())
print("\n")
time.sleep(2)
new_situation = np.array([[1,1,0]])
print("new situation",new_situation)
expected_output = sigmoid(np.dot(new_situation,weights))
print("Actual output of new situation: [1]")
print("predicted output of new situation:",expected_output)

