import praw
import pandas as pd

REDDIT = praw.Reddit(user_agent='redditflairdetector', client_id='PY-6WwMrA9O48Q', client_secret='rwwa13TTlmWYSeD8D9_kW13r6UE')

TOPICS = {"FLAIR":[], "COMMENT_NOS": [], "TITLE":[], "SCORE":[], "URL":[]}

FLAIRS = ["Food", "Non-Political", "AskIndia", "[R]eddiquette", "Business/Finance", "Scheduled", "Science/Technology", "Politics", "AMA", "Policy/Economy", "Photography", "Sports"]

SUBREDDIT = REDDIT.subreddit('india')

for FL in FLAIRS:
	GSRD = SUBREDDIT.search(FL, limit=100)
	for QRY in GSRD:
		TOPICS["FLAIR"].append(FL)
		TOPICS["TITLE"].append(QRY.title)
		TOPICS["SCORE"].append(QRY.score)
		TOPICS["URL"].append(QRY.url)
		TOPICS["COMMENT_NOS"].append(QRY.num_comments)

TDATA = pd.DataFrame(TOPICS)
TDATA = TDATA[(TDATA != 0).all(1)]
TDATA['COMBINED'] = TDATA["TITLE"] + " " + TDATA["URL"]
TDATA['RATIO'] = ((TDATA.COMMENT_NOS/TDATA.SCORE)+(TDATA.SCORE/TDATA.COMMENT_NOS))/2
TDATA['RATIO'] = TDATA['RATIO'].astype(int)
TDATA.to_csv('Data.csv', index=False)