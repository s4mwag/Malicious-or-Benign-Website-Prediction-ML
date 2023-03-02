import enum

class Models(enum.Enum):
    KNN = "KNN"
    RF = "Random Forest Classifier"
    SVM = "SVM"
    DT = "Decision Tree Classifier"
    XGB = "XGBoost"
    NN = "Neural Network"
    ALL = "All the models"
    
ifPreprocessing = False #Selecting features with GA !!! will update subset.csv !!!

ActiveModel = Models.ALL #Selects global model for the step below

ifOptimizing = False #Best k-value, optimal tree params etc, will only print to console
ifPredicting = True #Runs prediction, prints confusion matrix


subsets = ["subset0169.csv","subset0361.csv","subset2309.csv","subset5432.csv","subset5463.csv"
           ,"subset5661.csv","subset7534.csv","subset9948.csv","subset4706.csv","subset6396.csv"]
subsetsNumbers = [169,361,2309,5432,5463,5661,7534,9948,4706,6396]