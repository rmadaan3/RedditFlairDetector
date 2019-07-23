import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

def train_test(X,y):
 
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 7)

	print("\nResults of Logistic Regression")
	logisticreg(X_train, X_test, y_train, y_test)

	print("\nResults of Linear Support Vector Machine")
	linear_svm(X_train, X_test, y_train, y_test)
	
	print("\nResults of Random Forest")
	randomforest(X_train, X_test, y_train, y_test)

def linear_svm(X_train, X_test, y_train, y_test):

	LSVM = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=0.0001, random_state=42, max_iter=150, tol=None))])
	
	LSVM.fit(X_train, y_train)

	y_pred = LSVM.predict(X_test)

	print('Accuracy '+ str(round(accuracy_score(y_pred, y_test)*100,2)) + '%')

def logisticreg(X_train, X_test, y_train, y_test):

	LR = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression(n_jobs=1, C=1e5))])
	
	LR.fit(X_train, y_train)

	y_pred = LR.predict(X_test)

	print('Accuracy '+ str(round(accuracy_score(y_pred, y_test)*100,2)) + '%')

def randomforest(X_train, X_test, y_train, y_test):

	RF = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', RandomForestClassifier(n_estimators = 1000, random_state = 42))])
	
	RF.fit(X_train, y_train)

	y_pred = RF.predict(X_test)

	print('Accuracy '+ str(round(accuracy_score(y_pred, y_test)*100,2)) + '%')

if __name__ == "__main__":

	warnings.filterwarnings("ignore")

	TDATA = pd.read_csv('Data.csv')

	print("\nFlair Detection with Title as Feature:")
	train_test(TDATA.TITLE,TDATA.FLAIR)

	print("\nFlair Detection with URL as Feature:")
	train_test(TDATA.URL,TDATA.FLAIR)

	print("\nFlair Detection with Title & URL Combined as Feature:")
	train_test(TDATA.COMBINED,TDATA.FLAIR)
