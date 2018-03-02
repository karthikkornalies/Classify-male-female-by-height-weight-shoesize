#import enssential dependencies from sklearn
from  sklearn import neighbors
from sklearn import tree
from sklearn import ensemble

#[height, weight, shoe size]
X = [[181,80,44], [177, 70, 43], [160, 60, 38], [154, 54,37], 
	 [166,65,40], [190,90,47], [175,64,39], [177,70,40], [159, 55,39], [171,75,42], [181,85,43]]

#gender
Y =  ['male', 'female', 'female', 'female', 'male', 'male',
	  'male', 'female','male', 'female', 'male']

#create variables with classifiers
tree = tree.DecisionTreeClassifier()
neighbors = neighbors.KNeighborsClassifier()
randomForest = ensemble.RandomForestClassifier()

#fit them
tree = tree.fit(X,Y)
neighbors = neighbors.fit(X,Y)
randomForest = randomForest.fit(X,Y)

#create prediction variables for results
predictionForTree = tree.predict([[167,63,41]])
predictionForNeighbors = neighbors.predict([[167,63,41]])
predictionForRandomForest = randomForest.predict([[167,63,41]])

#print all the results
print("DecisionTree: ", predictionForTree)
print("NearestNeighbors: ", predictionForNeighbors)
print("RandomForest: ", predictionForRandomForest)
