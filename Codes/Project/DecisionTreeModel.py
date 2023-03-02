from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def dtOptimization(X_train, y_train):
    print("DT Parameter Optimization")
    params = { #Can set list(range(1, 10)) to get range for each param 
        "criterion" : ["entropy"], #Choices are "gini", "entropy" and "log_loss" where entropy wins standalone
        "max_leaf_nodes" : [5], #1st iteration: 5 best, 2nd iteration: 5 best 
        "min_samples_split" : [1, 10, 50], #1st iteration: 10 best, 2nd: 50 best, 3rd: 1 best (in combination w. others)
        "min_samples_leaf": [5], #1st iteration: 5 best, 2nd iteration: 1 best 
        "max_depth": [5, 10] #1st iteration: 10 best, 2nd iteration: 5 best 
        }
    
    optimal_params = GridSearchCV(DecisionTreeClassifier(random_state=12345), params, verbose=1, cv=5)
    """
    ### RESULTS ###
    Best Score:
    0.9974571888572579
    Best params:
    {'criterion': 'entropy', 'max_depth': 5, 'max_leaf_nodes': 5, 'min_samples_leaf': 1, 'min_samples_split': 1}
    """
    optimal_params.fit(X_train, y_train)

    print("Best Score: ")
    print(optimal_params.best_score_)
    print("Best params: ")
    print(optimal_params.best_params_)

def prediction(X_train, y_train, X_test):
    #n_estimators is the amount of trees in the forest, criterion determines the optimum mathematical split 
    classifier = DecisionTreeClassifier(criterion="entropy", max_depth=7, max_leaf_nodes=5)

    classifier.fit(X_train, y_train) 

    predictionDT = classifier.predict(X_test)

    return predictionDT