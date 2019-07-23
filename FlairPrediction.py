import warnings
import sklearn
import pickle
import praw
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

def detect_flair(id):
	DATA = {}

	SUBMISSION = REDDIT.submission(id=id)

	DATA['TITLE'] = SUBMISSION.title
	DATA['URL'] = SUBMISSION.url

	DATA['COMBINED'] = DATA['TITLE'] + ' ' + DATA['URL']
	
	return LOADED_MODEL.predict([DATA['COMBINED']])

warnings.filterwarnings("ignore")

TDATA = pd.read_csv('Data.csv')

X_train, x_test, y_train, y_test = train_test_split(TDATA.COMBINED,TDATA.FLAIR, test_size=0.3, random_state = 7)

LSVM = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=0.0001, random_state=42, max_iter=150, tol=None))])

LSVM.fit(X_train, y_train)

REDDIT = praw.Reddit(user_agent='redditflairdetector', client_id='PY-6WwMrA9O48Q', client_secret='rwwa13TTlmWYSeD8D9_kW13r6UE')

SUBREDDIT = REDDIT.subreddit('india')

SAVED_MODEL = pickle.dumps(LSVM)
LOADED_MODEL = pickle.loads(SAVED_MODEL)

print("Enter the number of queries to check in top category of the subreddit: ")
QUERIES = int(input())
for submission in SUBREDDIT.top(limit=QUERIES):
    print(detect_flair(submission.id),'\n',"Link: www.reddit.com"+submission.permalink)