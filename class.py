import pickle
from sklearn.feature_extraction import DictVectorizer
feats = pickle.load( open('bass.X.pkl', 'rb') )
print(feats[0])
v = DictVectorizer().fit(feats)
X = v.transform(feats)
print(X)

y = pickle.load( open('bass.y.pkl', 'rb') )
print(len(y))
print(y[:5])
majority = sorted(y[:900])[450]
baseline = (y[:900]==majority).mean()
print(majority, baseline)
majority = sorted(y[900:])[50]
baseline = (y[900:]==majority).mean()
print(majority, baseline)

#decision tree accuracy
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X[:900].toarray(), y[:900]) # See below *
y_pred = clf.predict(X[900:].toarray()) # See below *
acc = (y_pred==y[900:]).mean() #how many points are correct (sum/total)
print(acc)

#SVM accuracy
from sklearn.svm import LinearSVC
clf = LinearSVC().fit(X[:900], y[:900])
y_pred = clf.predict(X[900:])
acc = (y_pred==y[900:]).mean()
print(acc)

#nearest neighbor accuracy
from sklearn.neighbors import KNeighborsClassifier 
clf = KNeighborsClassifier().fit(X[:900], y[:900]) 
y_pred = clf.predict(X[900:]) # *
acc = (y_pred==y[900:]).mean()
print(acc)

#K fold cross validation
from sklearn.model_selection import cross_val_score 
acc_ = cross_val_score(LinearSVC(), X, y, cv=10)
print(acc_.mean())
print(acc_.std())

#confusion matrix and stratified k fold validation
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold 
import matplotlib.pyplot as plt 
from sklearn.metrics import ConfusionMatrixDisplay

clf = LinearSVC()
skf = StratifiedKFold(n_splits=3)
conf_mats = [ confusion_matrix( y[test], clf.fit(X[train], y[train]).predict(X[test]) ) 
                                                    for train, test in skf.split(X, y)] 

for cm in conf_mats: 
    print (cm)
