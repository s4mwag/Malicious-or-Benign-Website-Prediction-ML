#!/bin/python3

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from TypesAndConstants import *
import GAFeatureSetSelection as ga
import RemoveFeatures as rf
import KNNModel as knn
import AccuracyCalculator as ac
import SVMmodel as SVM
import XGBoostModel as xgb
import DecisionTreeModel as dt
import RandomForestModel as rfm
import NeuralNetworkModel as nnm
import csv  

X_train, X_test = None, None

def runClassification():
    global X_train, X_test
    data = pd.read_csv("../data.csv", encoding="ISO-8859-1", on_bad_lines='skip', sep=',') 
    data.drop("url", axis=1, inplace=True)
    # Save independent variables in "X"
    X = data.drop("label", axis=1)
    X = X.values
    # Save dependent variable in "y"
    y = data["label"]
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=12345)
    ################################################################################################################################
    #If we are preprocessing, extract feature selection from genetic algorithm
    if ifPreprocessing == True:
        normalization()
        features = ga.gaFeatureSelected(X_train, y_train)
        print(features)

        #Extract subset of features based on GA selected features. 
        #A subset.csv is created here to use with different models
        rf.RemoveFeatures(data, features)

        #Load dataset with removed features, create identical train/test split
        #data = pd.read_csv("subset.csv", encoding="ISO-8859-1", on_bad_lines='skip', sep=',') 
        #X = data.values
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12345)
    ################################################################################################################################
    if ifOptimizing == True:
        #Load dataset with removed features, create identical train/test split
        data = pd.read_csv("subsets/subset.csv", encoding="ISO-8859-1", on_bad_lines='skip', sep=',') #choose subset file manually 
        X = data.values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=12345)
        
        if ActiveModel == Models.KNN or ActiveModel == Models.ALL:
            knn.knnOptimization(X_train, y_train) 
        if ActiveModel == Models.RF:
            rfm.rfOptimization(X_train, y_train)
        if ActiveModel == Models.SVM or ActiveModel == Models.ALL:
            #normalization()
            #standardization()
            SVM.svmOptimization(X_train, y_train)
        if ActiveModel == Models.DT or ActiveModel == Models.ALL:
            dt.dtOptimization(X_train, y_train)
        if ActiveModel == Models.XGB or ActiveModel == Models.ALL:
            xgb.xgbOptimization(X_train, y_train)
        if ActiveModel == Models.NN or ActiveModel == Models.ALL:
            nnm.NNOptimizeParameters(X_train, y_train)

    ################################################################################################################################
    if ifPredicting == True:
        for i in range(10):           
            data = pd.read_csv("../data.csv", encoding="ISO-8859-1", on_bad_lines='skip', sep=',') 
            data.drop("url", axis=1, inplace=True)
            # Save independent variables in "X"
            X = data.drop("label", axis=1)
            X = X.values
            # Save dependent variable in "y"
            y = data["label"]
            y = y.values
            X_train, X_test, y_train, y_test = train_test_split(
    	        X, y, test_size=0.2, random_state=subsetsNumbers[i])
            #Load dataset with removed features, create identical train/test split             
            data = pd.read_csv(subsets[i], encoding="ISO-8859-1", on_bad_lines='skip', sep=',')             
            X = data.values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=subsetsNumbers[i])            
            if ActiveModel == Models.KNN or ActiveModel == Models.ALL:
                normalization()
                standardization()
                predictionKNN = knn.prediction(X_train, y_train, X_test)
                #Calculate the accuracy for KNN (Will print True Positives and Overall Accuracy)
                print("-----KNN")         
                printToAccuracyFile(subsets[i],Models.KNN._name_)       
                ac.accuracyCalc(y_test,predictionKNN)

            if ActiveModel == Models.RF or ActiveModel == Models.ALL:
                predictionRF = rfm.prediction(X_train, y_train, X_test)
                #Calculate the accuracy for Random Forest (Will print True Positives and Overall Accuracy)
                print("-----Random Forest Classifier")
                printToAccuracyFile(subsets[i],Models.RF._name_)  
                ac.accuracyCalc(y_test,predictionRF)

            if ActiveModel == Models.SVM or ActiveModel == Models.ALL:
                normalization()
                standardization()
                predictionSVM = SVM.prediction(X_train, y_train, X_test)
                #Calculate the accuracy for SVM (Will print True Positives and Overall Accuracy)                
                print("-----SVM with Radial Basis Function (RBF)")
                printToAccuracyFile(subsets[i],Models.SVM._name_)  
                ac.accuracyCalc(y_test,predictionSVM)

            if ActiveModel == Models.DT or ActiveModel == Models.ALL:
                predictionDT = dt.prediction(X_train, y_train, X_test)
                print("-----Decision Tree Classifier")
                printToAccuracyFile(subsets[i],Models.DT._name_)  
                ac.accuracyCalc(y_test,predictionDT)

            if ActiveModel == Models.XGB or ActiveModel == Models.ALL:
                normalization()
                standardization()
                predictionXGBoost = xgb.prediction(X_train, y_train, X_test, y_test)
                print("-----eXtreme Gradient Boosting")
                printToAccuracyFile(subsets[i],Models.XGB._name_)  
                ac.accuracyCalc(y_test,predictionXGBoost)

            if ActiveModel == Models.NN or ActiveModel == Models.ALL:
                print("-----Neural Network")
                # normalization()
                # standardization()
                predictionNN = nnm.prediction(X_train, y_train, X_test,len(data.columns))
                printToAccuracyFile(subsets[i],Models.NN._name_)  
                ac.accuracyCalc(y_test,predictionNN)

def normalization():
    global X_train, X_test
    X_train = preprocessing.normalize(X_train)
    X_test = preprocessing.normalize(X_test)

def standardization():
    global X_train, X_test
    scaler = preprocessing.StandardScaler()
    X_test = scaler.fit_transform(X_test)
    X_train = scaler.fit_transform(X_train)

def printToAccuracyFile(currentSubset, currentModel):
    with open('ConfusionMatrices.csv', mode='a') as file_:
        file_.write("{},{}".format(currentSubset, currentModel))
        file_.write("\n")  # Next line.



def main():
    runClassification()

if __name__ == '__main__':
    main()