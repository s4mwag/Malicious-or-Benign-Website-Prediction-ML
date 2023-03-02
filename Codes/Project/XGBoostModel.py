from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pickle #Used for saving and loading scikit models

### Optimizing for best parameters ###
#1st iteration:
#Settings: {"max_depth" : [3, 5, 10], "learning_rate" : [0.1, 1, 10], "gamma" : [0, 0.25, 1.0], "reg_lambda" : [0, 1.0, 10.0], "scale_pos_weight" : [1, 3, 5], 'n_estimators': [50, 100]}, verbose=1, n_jobs=1, cv=3)
#Results: {'gamma': 1.0, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'reg_lambda': 0, 'scale_pos_weight': 1}
#overall: 0.997643263735838

#2nd iteration:
#Settings: {"max_depth" : [1, 2, 3], "learning_rate" : [0.001, 0.01, 0.1], "gamma" : [1.0, 5.0, 10.0], "reg_lambda" : [0], "scale_pos_weight" : [0.0, 0.5, 1.0], 'n_estimators': [25, 50]}, verbose=1, n_jobs=1, cv=10)
#Results: {'gamma': 1.0, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'reg_lambda': 0, 'scale_pos_weight': 1.0}
#Overall:  0.9976277459037577
#Comment: Overall was a little bit worse. Result params are the same.

def xgbOptimization(X_train, y_train):
    print("XGB Parameter Optimization")
    XGBSavedmodel = XGBClassifier(objective="binary:logistic") #Problem is classification (binary)
    #Range of params to test with
    optimal_params = GridSearchCV(XGBSavedmodel,
                    {"max_depth" : [1, 2, 3], 
                     "learning_rate" : [0.001, 0.01, 0.1], 
                     "gamma" : [1.0, 5.0, 10.0], 
                     "reg_lambda" : [0], 
                     "scale_pos_weight" : [0.0, 0.5, 1.0],
                     "n_estimators": [25, 50]}, verbose=1, n_jobs=1, cv=10) 
    
    optimal_params.fit(X_train, y_train)
    print("Best Score: ")
    print(optimal_params.best_score_)
    print("Best params: ")
    print(optimal_params.best_params_)

def prediction(X_train, y_train, X_test, y_test):

    #Use a logistic regression approach to classify data with params gotten from optimization
    #OLD: XGBSavedmodel = XGBClassifier(eval_metric="aucpr", objective="binary:logistic", gamma=1.0, learning_rate=0.1, max_depth=3, n_estimators=50, reg_lambda=0, scale_pos_weight=1)
    XGBSavedmodel = XGBClassifier(eval_metric="aucpr", objective="binary:logistic", learning_rate=0.1, max_depth=3, n_estimators=50)

    #Stop early if no improvement after X rounds, evaluate training data based on test data, evaulation through Area under the PR curve.
    XGBSavedmodel.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    file = "models/xgb.pickle" #Filename and location

    #pickle.dump(XGBSavedmodel, open(file, "wb")) #Save binary representation of model

    #XGBLoadedModel = pickle.load(open(file, "rb")) #Load model

    XGBpredictions = XGBSavedmodel.predict(X_test) #Predict base on model

    return XGBpredictions