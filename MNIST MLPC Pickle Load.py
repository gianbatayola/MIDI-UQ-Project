import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, accuracy_score
import pickle
from numpy import var



matrix = pickle.load(open('MLPC Bagging Ensemble Matrix', 'rb'))

model = pickle.load(open('MLPC Bagging Ensemble Model', 'rb'))

MNIST = pd.read_csv('train.csv')

features = MNIST.iloc[:5000,1:].values.reshape((5000,-1))
labels = MNIST.iloc[:5000,0].values.reshape(5000)

train,test,train_labels,test_labels = train_test_split(features,
                                                       labels,
                                                       test_size = .30,
                                                       random_state = 0,
                                                       shuffle = True)

clf_list = []
evs_list = []
var_list = []
for i in range(10):
    clf = model.estimators_[i]
    predictions = clf.predict(test)
    clf_list.append(accuracy_score(test_labels,predictions))
    evs_list.append(explained_variance_score(test_labels,predictions))
    var_list.append(var(predictions-test_labels))
    

print(evs_list)
predictions = model.predict(test)
print(explained_variance_score(test_labels,predictions))

print()
print(var_list)
print(var(predictions-test_labels))



                    

