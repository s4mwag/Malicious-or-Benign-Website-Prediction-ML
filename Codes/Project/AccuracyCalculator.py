from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from TypesAndConstants import *
import numpy as np
import matplotlib.pyplot as plt # new
import pandas as pd

np.set_printoptions(precision=4, suppress=True)

def accuracyCalc(y_test, prediction):
    confusionMatrix = confusion_matrix(y_test, prediction, normalize='true')    
    with open('plots/ConfusionMatrices.csv', mode='a') as file_:
        file_.write("{}".format(confusionMatrix[0][0]))
        file_.write("\n")
        file_.write("{}".format(confusionMatrix[1][1]))
        file_.write("\n")     
        file_.write("{}".format(confusionMatrix[0][1]))
        file_.write("\n") 
        file_.write("{}".format(confusionMatrix[1][0]))
        file_.write("\n") 
    ConfutionMatrixDiagonal = np.diag(confusionMatrix)
    FP = confusionMatrix.sum(axis=0) - ConfutionMatrixDiagonal  
    FN = confusionMatrix.sum(axis=1) - ConfutionMatrixDiagonal
    TP = ConfutionMatrixDiagonal
    TN = confusionMatrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    print("Confusion Matrix:")
    print(confusionMatrix)
    print("True Positive Rate Malicious:", round(TPR[0], 4))
    print("True Positive Rate Benign:", round(TPR[1], 4))

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print("Overall Accuracy: ", round(ACC[0], 4))

    # #Plotting confusion matrix based on malicious point of view as .pdfs
    # confusion_M = confusion_matrix(y_test, prediction)
    # confusion_M = confusion_M.astype('float') / confusion_M.sum(axis=1)[:, np.newaxis]
    # cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_M, display_labels = ["Malicious", "Benign"])

    # cm_display.plot()
    # plt.title("Confusion Matrix: " + ActiveModel._name_)
    # plt.show()
    # plt.savefig("plots/" + ActiveModel._name_ + ".pdf") 
