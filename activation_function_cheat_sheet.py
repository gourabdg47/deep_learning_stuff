import numpy as np

#np.exp() = 2.718281 = 'Euler's number', so np.exp(2) is 2.718281^2
logits = [2.0, 1.0, 0.1]

###################################################################

# softmax
exps = [np.exp(i) for i in logits]
softmax = [j/sum(exps) for j in exps]
print("Softmax: ",softmax) # sum of the total output is always 1.0
print("Sum of softmax: ",sum(softmax))

# sigmoid 

def sigmoid(x):
    return 1/(1 + np.exp(-x))

sigmoid = [sigmoid(k) for k in logits]
print("Sigmoid: ",sigmoid) # output always within the range of [0,1]

# tanh

def tanh(x):
    return (2/1 + np.exp(-2*x)) - 1

tanh = [tanh(x) for x in logits]
print("Tanh: ",tanh) # values will always lie in range of [-1,1]
# tanh is a scaled sigmoid function 2 * sigmoid(2x) - 1

# relu

def relu(x):
    return max(0,x)

relu = [relu(x) for x in logits]
print("Relu: ",relu) #values always ranges from [0,infinity]

################################################################
