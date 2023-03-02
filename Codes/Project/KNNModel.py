from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

#Determine the best amount of neighbors to use in prediction

def knnOptimization(X_train, y_train):
    print("KNN Parameter Optimization")
    parameters = {
        "n_neighbors": [2, 3, 8, 12],
        "weights" : ["uniform", "distance"],
        "algorithm" : ["ball_tree", "kd_tree", "brute"],
        "leaf_size" : [20, 30, 40],
        "p" : [1, 2, 3],
        "n_jobs" : [-1]
        }
    gridSearch = GridSearchCV(KNeighborsClassifier(), parameters, verbose=1, cv=5)
    gridSearch.fit(X_train, y_train)

    """
    Best Score 1st round: 
    0.9966354473725805
    Best params: 
    {'algorithm': 'brute', 'leaf_size': 20, 'n_jobs': -1, 'n_neighbors': 8, 'p': 1, 'weights': 'distance'}
    """

    print("Best Score: ")
    print(gridSearch.best_score_)
    print("Best params: ")
    print(gridSearch.best_params_)

def prediction(X_train, y_train, X_test):
    # p = 1 = manhattan. p = 2 = euclidean. neighbors are derived from the latest test run which is based on the latest subset
    knn_model_classified = KNeighborsClassifier(leaf_size=20, n_neighbors=3, p=1, weights="uniform") #Before change: n_neighbors=3, metric = 'minkowski', p = 2

    knn_model_classified.fit(X_train, y_train) 

    predictionKNN = knn_model_classified.predict(X_test)

    return predictionKNN