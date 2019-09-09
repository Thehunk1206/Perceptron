import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import time

#style.use('fivethirtyeight')
fig = plt.figure()
graph1 = fig.add_subplot(1,1,1)
plt.xlabel("Iterations")
plt.ylabel("Error")

error_lst = []



def sigmoid(x,derivative=False):
    if derivative==True:
        return x*(1-x)
    return 1/(1+ np.exp(-x))

#data
training_inputs = np.array([[1,0,1],
                            [1,1,1],
                            [1,0,1],
                            [1,0,1],
                            [0,1,0],
                            [1,1,1],
                            [1,0,0],])
training_outputs = np.array([[0,1,0,0,1,1,1]]).T


#initializing random weights
np.random.seed(0)
weights = 2*np.random.random((3,1))
print('Initializing weights')
print(weights)
print("Starting traing process\n")



#training
for i in range(20000):

    input_layer = training_inputs

    output = sigmoid(np.dot(input_layer,weights))

    error = training_outputs - output

    adjustments = error * sigmoid(output,derivative=True)

    if i%100 == 0:
        graph1.scatter(i,error[1],color='b')
        #plt.pause(0.02)
        print(error.flatten())

    weights+=np.dot(input_layer.T,adjustments)
#plt.show() 


print("===========================================")
print("weights after training")

print(weights,"\n")

print("===========================================")
print("output after training\n")

print("Actual output",training_outputs.flatten(),"\n")
print("Output by perceptron: ",output.flatten())
print("\n")

new_situation = np.array([[1,0,1]])
print("new situation",new_situation)
expected_output = sigmoid(np.dot(new_situation,weights))
print("Actual output of new situation: ")
print("predicted output of new situation:",expected_output)

#graph1.plot(error_lst)
