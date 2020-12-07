import numpy as np
from Utils import ArtificialNeuralNetwork
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


def ANNTrain(filename):
    # result is normally between 0.55 - 0.75 but may fluctuate
    input = np.genfromtxt(filename, delimiter=",", skip_header=1)
    xtrain = input[:730,:8]
    ytrain = input[:730, 8:]
    
    xtest = input[730:,:8]
    ytest = input[730:, 8:]
  
    
    # scaler = StandardScaler()
    # scaler.fit(xtrain)
    # X_train = scaler.transform(xtrain)
    # X_test = scaler.transform(xtest)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(xtrain)
    X_test = scaler.fit_transform(xtest)

    print("Early Stopping = False")
    mlp = MLPRegressor(solver="lbfgs", activation='relu', learning_rate_init=0.01, hidden_layer_sizes=(7), early_stopping = False, max_iter = 120000, 
        learning_rate = 'invscaling', alpha=0.2)


    max1 = 0
    print("Looping through 30 tries... ")
    for i in range(30):
        mlp.fit(X_train,ytrain.ravel())
        # print(i, " ", mlp.score(X_test, ytest))
        if mlp.score(X_test, ytest) > max1:
            max1 = mlp.score(X_test, ytest)
    print("Max Score: ", max1)
    
    print()
    # Early Stopping prevents overfitting
    print("Early Stopping = True")

    mlp = MLPRegressor(solver="lbfgs", activation='relu', learning_rate_init=0.01, hidden_layer_sizes=(7), early_stopping = True, max_iter = 120000, 
        learning_rate = 'invscaling', alpha=0.2, validation_fraction=0.01)


    max2 = 0
    print("Looping through 30 tries... ")
    for i in range(30):
        mlp.fit(X_train,ytrain.ravel())
        # print(i, " ", mlp.score(X_test, ytest))
        if mlp.score(X_test, ytest) > max2:
            max2 = mlp.score(X_test, ytest)
    print("Max Score: ", max2)
    
    print()
    if max1 > max2:
        print("Without Early stopping better!")
    elif max2 > max1:
        print("With Early Stopping Better!")
    else:
        print("Results About the Same!")

    # To make the results better, I could have shuffled the data instead of choosing the first 730 datapoints to train right away.
ANNTrain("Concrete_Data.csv")
