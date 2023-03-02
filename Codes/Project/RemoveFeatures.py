import pandas as pd
def RemoveFeatures(data, features):
    
	#Below we are creating the new subset from feature selection by dropping "bad" features
	data.drop("label", axis=1, inplace=True)
	
	#Save indexes of all "bad features"
	delIndex = []
	for i in range(len(features)-1):
		if features[i] == False:
			delIndex.append(i)
	
	print("Before drop")
	print(data.columns)
	data.drop(data.columns[delIndex], axis=1, inplace=True) #Dropping the "bad" features
	data.to_csv("subset2309.csv", index=False)
	print("After drop")
	print(data.columns)

