"""
Author : Loris Wintjens
Goal : Detect email spam using ML
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# Load data set
data = pd.read_csv('spam.csv')


# split in to training and test data
X, Y = data["EmailText"], data["Label"]

X_train, Y_train = X[0:4457], Y[0:4457]
X_test, Y_test = X[4457:],Y[4457:]

# extract features
cv = CountVectorizer()
features = cv.fit_transform(X_train)

# build a model
tunedParameters = {'kernel': ['linear','rbf'], 'gamma': [1e-3,1e-4], 'C': [1,10,100,1000]}
model = GridSearchCV(svm.SVC(),tunedParameters)
model.fit(features,Y_train)
print(model.best_params_)
# test accuracy
features_test = cv.transform(X_test)

print('Accuracy of the model is : ', model.score(features_test,Y_test))