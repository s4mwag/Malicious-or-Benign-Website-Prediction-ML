from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.callbacks import ConsecutiveStopping
from sklearn_genetic.plots import plot_fitness_evolution
import matplotlib.pyplot as plt
from sklearn import neighbors, linear_model

def gaFeatureSelected(X_train, y_train):
    	
	clf = neighbors.KNeighborsClassifier(n_neighbors=3, metric = 'minkowski', p = 1)
	callback = ConsecutiveStopping(generations=3, metric='fitness')

	evolved_estimator = GAFeatureSelectionCV(
		estimator=clf, #Use KNN for classification
		cv=5,
		scoring="accuracy", #Score based accuracy
		population_size=25, #What is the optimal population size
		generations=50, #The amount of iterations
		n_jobs=-1, #-1 means we use all cores
		verbose=True, #Print output
		keep_top_k=3, #Number of best solutions to keep in the hof object.
		elitism=True, #Takes the Tournament Size (default=3) best solutions to the next generation. 
		crossover_probability=0.9, 
		mutation_probability=0.05
	)
	evolved_estimator.fit(X_train, y_train,callbacks=callback)

	#Save the GA fitness evolution as a .pdf
	plot_fitness_evolution(evolved_estimator)

	plt.savefig("plots/GAfitnessEvolution2309.pdf")
	
	return evolved_estimator.best_features_
    
