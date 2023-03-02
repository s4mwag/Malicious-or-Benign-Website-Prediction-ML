from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def rfOptimization(X_train, y_train):
    print("RF Parameter Optimization")
    params = { #Or use param range like:  list(range(1, 10))
        "n_estimators" : [90, 100, 110], #1st iteration: 100 best of [50, 100, 150]
        "criterion" : ["entropy"], #1st iteration: entropy best out of "gini", "entropy", "log_loss"
        "max_depth" : [19, 20, 21], #1st iteration: 20 best of [10, 20, 30]
        "min_samples_split" : [1, 2, 3], #1st iteration: 1 best
        "min_samples_leaf" : [1, 2, 3], #1st iteration: 1 best
        "min_weight_fraction_leaf" : [0.0, 0.2, 0.5], #1st iteration: bugged has to be between 0.0 and 0.5
        }
    
    optimal_params = GridSearchCV(RandomForestClassifier(random_state=12345), params, verbose=1, cv=5)
    """
    ### RESULTS ###
    Best Score 1st: 
    0.9978913394524304
    Best params 1st:
    {'criterion': 'entropy', 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 1, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100}


    Best Score 2nd: 
    0.9979068421264552
    Best params 2nd:
    {'criterion': 'entropy', 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 1, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 90}
    """
    optimal_params.fit(X_train, y_train)

    print("Best Score: ")
    print(optimal_params.best_score_)
    print("Best params: ")
    print(optimal_params.best_params_)

def prediction(X_train, y_train, X_test):
    #n_estimators is the amount of trees in the forest, criterion determines the optimum mathematical split 
    classifier = RandomForestClassifier(n_estimators=90, n_jobs=-1, max_depth=20, criterion="entropy")

    classifier.fit(X_train, y_train) 

    predictionRF = classifier.predict(X_test)

    return predictionRF