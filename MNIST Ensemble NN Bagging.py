import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import plot_confusion_matrix, accuracy_score
import pickle
import time

start_time = time.time()


MNIST = pd.read_csv('train.csv')

features = MNIST.iloc[:5000,1:].values.reshape((5000,-1))
labels = MNIST.iloc[:5000,0].values.reshape(5000)

train,test,train_labels,test_labels = train_test_split(features,
                                                       labels,
                                                       test_size = .30,
                                                       random_state = 0,
                                                       shuffle = True)


MLPC = MLPClassifier(hidden_layer_sizes = (300,300,300))

clf = BaggingClassifier(base_estimator = MLPC,
                        n_estimators = 10)
clf.fit(train,train_labels)
predictions = clf.predict(test)

disp = plot_confusion_matrix(clf,test,test_labels,values_format = '.3g')
title = "MLPC Ensemble Accuracy:" + ' ' + str(accuracy_score(test_labels,predictions))
disp = disp.ax_.set_title(title)

print("{} seconds".format(str(time.time() - start_time)))

pickle.dump(disp, open('MLPC Bagging Ensemble Matrix', 'wb'))
pickle.dump(clf, open('MLPC Bagging Ensemble Model', 'wb'))