import numpy as np
from Utils import ArtificialNeuralNetwork, mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

def d_mse(y, y_hat):
    return -(y- y_hat)



def d_relu(x):
    relu = np.maximum(x, 0)
    relu[relu<=0] = 0
    relu[relu>0] = 1
    return relu

def train(
    neural_network,
    training_inputs,
    training_labels,
    n_epochs,
    learning_rate=0.001
):
    losses = []
    m = training_inputs.shape[1]
    y_hat, a_memory, z_memory = neural_network.forward(training_inputs)
    losses.append(mean_squared_error(training_labels, y_hat))
    for e in range(n_epochs):
        
        dA_l = d_mse(training_labels, y_hat)
        for l in range(len(neural_network.layers), 0, -1):
            A_prev_l = a_memory[l-1]
            dZ_l = dA_l*d_relu(z_memory[l])
            dW_l = np.dot(dZ_l, A_prev_l.T)
            dB_l = (1/m) * np.sum(dZ_l, axis = 1, keepdims=True)
            dA_l = np.dot(neural_network.layers[l-1].T, dZ_l)
            neural_network.layers[l-1] = neural_network.layers[l-1] - learning_rate * dW_l
            neural_network.biases[l-1] = neural_network.biases[l-1] - learning_rate * dB_l
        y_hat, a_memory, z_memory = neural_network.forward(training_inputs)
        losses.append(mean_squared_error(training_labels, y_hat))
    return losses

