from sklearn import svm
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split

# load dataset for training
print('')
print('--> Loading dataset')
df = pd.read_csv('../data/data_rh.csv', index_col=0)
print('DONE')

# prepare X and y variables
X = np.array(df.iloc[:,:-1])
y = np.array(df['y'])
print('')
print('--> Summary')
print(f'Number of samples: {len(X)}')
print(f'Number of features: {X.shape[1]}')
print(f'Number of classes: {len(np.unique(y))}')

# train test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
print(f'Number of samples for Training: {len(X_train)}')
print(f'Number of samples for Test: {len(X_test)}')

# define and train SVM model
clf = svm.SVC(probability=True)
print('')
print('--> Training SVM Model')
clf.fit(X_train, y_train)
print('DONE')

#Predict the response for train & test datasets
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print('')
print('--> Model performances')
print("Accuracy Train:",metrics.accuracy_score(y_train, y_train_pred))
print("Accuracy Test: ",metrics.accuracy_score(y_test, y_test_pred))

# save the model to disk
print('')
print('--> Saving model in "../models/model_svm.sav"')
filename = '../models/model_svm.sav'
pickle.dump(clf, open(filename, 'wb'))
print("DONE")
print('')
