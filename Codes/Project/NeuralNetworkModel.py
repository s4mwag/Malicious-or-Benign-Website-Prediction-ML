from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def prediction(X_train, y_train, X_test,columnSize):    
    # Quick sanity check with the shapes of Training and Testing datasets # print(X_train.shape)# print(y_train.shape)# print(X_test.shape)# print(y_test.shape)
    # Here we gonna create a sequential layers of NN phases 
    # A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
    classifier = Sequential()
    # Defining the Input layer and FIRST hidden layer,both are same! #relu is a simple function that returns 0 for any negative input and the input itself for any positive input.
    #uniform is the algorithm has to decide the value for each weight in network computations. 
    classifier.add(Dense(units=10, input_dim=columnSize, kernel_initializer='uniform', activation='relu'))
    #Defining the SECOND hidden layer, here we have not defined input because it is second layer and it will get input as the output of first hidden layer
    classifier.add(Dense(units=10, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=10, kernel_initializer='uniform', activation='relu'))
    # classifier.add(Dense(units=10, kernel_initializer='uniform', activation='relu'))
    # Defining the Output layer # And output_dim will be equal to the number of factor levels
    # # The activation function fore binary classification is sigmoid  # Sigmoid: maps any input value to a value between 0 and 1. It is commonly used in binary classification
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    # adam is one of algorithm of SGG that keep updating weights # cross-entropy loss is used for binary (0 or 1) classification applications
    # binary_crossentropy used to calculate the accuracy the loss function to measure the accuracy #Computes the cross-entropy loss between true labels and predicted labels.
    # metrics== the way we will compare the accuracy after each step of SGD
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # fitting the Neural Network on the training data # how many rows go to network for each SSE calculation 
    classifier.fit(X_train,y_train, batch_size=15 , epochs=15, verbose=1)
    # Predictions on testing data
    Predictions=classifier.predict(X_test)
    for idx, pred in enumerate(Predictions):
        if pred > 0.5:
            Predictions[idx] = 1
        else:
            Predictions[idx] = 0
    return Predictions

###############################################  finding best hyperparameter using GridSearchCV  ###############################################
def createClassification(Optimizer_Trial, Neurons_Trial):
    # Creating the classifier ANN model
    classifier = Sequential()
    classifier.add(Dense(units=Neurons_Trial, input_dim=47, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=Neurons_Trial, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=Optimizer_Trial, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

def NNOptimizeParameters(X_train,y_train):
    Parameter_Trials={'batch_size':[10,15,20,30],
                      'epochs':[10,20],
                      'Optimizer_Trial':['adam', 'rmsprop'],
                      'Neurons_Trial': [5,10,20]
                     }
    # Creating the classifier ANN
    classifierModel=KerasClassifier(createClassification, verbose=0)
    # Creating the Grid search space
    # See different scoring methods by using sklearn.metrics.SCORERS.keys()
    grid_search=GridSearchCV(estimator=classifierModel, param_grid=Parameter_Trials, scoring='f1', cv=5)
    # Running Grid Search for different parameters
    grid_search.fit(X_train,y_train, verbose=1)
    # printing the best parameters
    print('\n#### Best hyperparameter ####')
    print(grid_search.best_params_)
