import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV

def svmOptimization(X_train, y_train):
    print("SVC Parameter optimization")
    params = {
        #"kernel" : ["linear", "poly", "rbf", "sigmoid", "precomputed"] #complains about n_splits > 2, 
        #"gamma" : [1, 0.1, 0.01, 0.001, 0.0001],
        #"penalty" : ["l1", "l2"],
        #"loss" : ["hinge", "squared_hinge"],
        #"tol" : [1e-5, 1e-4, 1e-3] #doesnt work
        #"C" : [0.1, 1.0, 1.5],
        #"max_iter" : [2000] #https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati
        }
    
    optimal_params = GridSearchCV(SVC(random_state=12345), params, verbose=1, cv=1)

    optimal_params.fit(X_train, y_train)

    print("Best Score: ")
    print(optimal_params.best_score_)
    print("Best params: ")
    print(optimal_params.best_params_)

def prediction(X_train, y_train, X_test):
    
    #Creating and fitting data to a linear SVM model
    svcClassifierLinear = LinearSVC(dual=False)
    svcClassifierLinear.fit(X_train, y_train)
    #Creating and fitting data to a Radial Basis Function (RBF) SVM model
    svcClassifierRBF = SVC(kernel='rbf')
    svcClassifierRBF.fit(X_train, y_train)
    
    """
    #Score using a linear kernel
    print("Score Linear")
    print(svcClassifierLinear.score(X_train, y_train))

    #Score using a RBF kernel
    print("Score RBF")
    print(svcClassifierRBF.score(X_train, y_train))
    
    #Prediction using a linear kernel
    y_pred_Linear = svcClassifierLinear.predict(X_test)
    """
    #Prediction using a RBF kernel
    y_pred_RBF = svcClassifierRBF.predict(X_test)
    return y_pred_RBF