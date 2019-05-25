"""Create a end point for boot identification solution."""
import os
from flask import Flask
from flask_restful import reqparse, Api, Resource
import pickle
from model import RFModel
import pandas as pd
import json
import datetime as dt
import difflib
from textblob import TextBlob
import sys
import numpy as np

app = Flask(__name__)
api = Api(app)

model = RFModel()

clf_path = 'lib/models/RandomForestClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

clean_data_path = 'lib/models/CleanData.pkl'
with open(clean_data_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()

parser.add_argument('banned_by')
parser.add_argument('no_follow')
parser.add_argument('link_id')
parser.add_argument('gilded')
parser.add_argument('author')
parser.add_argument('author_verified')
parser.add_argument('author_comment_karma')
parser.add_argument('author_link_karma')
parser.add_argument('num_comments')
parser.add_argument('created_utc')
parser.add_argument('score')
parser.add_argument('over_18')
parser.add_argument('body')
parser.add_argument('downs')
parser.add_argument('is_submitter')
parser.add_argument('num_reports')
parser.add_argument('controversiality')
parser.add_argument('quarantine')
parser.add_argument('ups')
parser.add_argument('is_bot')
parser.add_argument('is_troll')
parser.add_argument('recent_comments')

class Botidentification(Resource):
    """Class for add end points."""

    def get(self):
        """HTTP GET /.

        Test the server.
        """
        return {'Status': 'The server run'}

    def post(self):
        """HTTP POST /.

        Boot identification solution end point.

        Returns
        -------
        Return if the Form Data is about a Normal user, Bot or a Troll.

        """
        app.logger.info("Received request")
        
        # Use parser and find the user's query
        args = parser.parse_args()
        
        no_follow = bool(args['no_follow'])
        created_utc = int(args['created_utc'])
        author_verified = bool(args['author_verified'])
        author_comment_karma = float(args['author_comment_karma'])
        author_link_karma = float(args['author_link_karma'])
        over_18 = bool(args['over_18'])
        is_submitter = bool(args['is_submitter'])
        body = str(args['body'])
        recent_comments = str(args['recent_comments'])
        recent_num_comments = 0
        recent_num_last_30_days = 0
        recent_avg_no_follow = 0
        recent_avg_gilded = 0
        recent_avg_responses = 0
        recent_percent_neg_score = 0
        recent_avg_score = 0
        recent_min_score = 0
        recent_avg_controversiality = 0
        recent_avg_ups = 0
        recent_avg_diff_ratio = 0
        recent_max_diff_ratio = 0
        recent_avg_sentiment_polarity = 0
        recent_min_sentiment_polarity = 0
        
        created_utc = pd.to_datetime(created_utc, unit='s')
        
        def diff_ratio(_a, _b):
            return difflib.SequenceMatcher(a=_a,b=_b).ratio()

        def last_30(a, b):
            return a - dt.timedelta(days=30) < pd.to_datetime(b, unit='s')
        
        recent_comments = pd.read_json(recent_comments, dtype={
            "banned_by": str,
            "no_follow": bool,
            "link_id": str,
            "gilded": np.float64,
            "author": str,
            "author_verified": bool,
            "author_comment_karma": np.float64,
            "author_link_karma": np.float64,
            "num_comments": np.float64,
            "created_utc": np.float64,
            "score": np.float64,
            "over_18": bool,
            "body": str,
            "downs": np.float64,
            "is_submitter": bool,
            "num_reports": np.float64,
            "controversiality": np.float64,
            "quarantine": bool,
            "ups": np.float64})
        if(len(recent_comments) > 0):
            recent_num_comments = len(recent_comments)
            recent_num_last_30_days = recent_comments['created_utc'].apply(lambda x: last_30(created_utc, x)).sum()
            recent_avg_no_follow = recent_comments['no_follow'].mean()
            recent_avg_gilded = recent_comments['gilded'].mean()
            recent_avg_responses = recent_comments['num_comments'].mean()
            recent_percent_neg_score = recent_comments['score'].apply(lambda x: x < 0).mean() * 100
            recent_avg_score = recent_comments['score'].mean()
            recent_min_score = recent_comments['score'].min()
            recent_avg_controversiality = recent_comments['controversiality'].mean()
            recent_avg_ups = recent_comments['ups'].mean()
            diff = recent_comments['body'].str.slice(stop=200).fillna('').apply(lambda x: diff_ratio(body, x))
            recent_avg_diff_ratio = diff.mean()
            recent_max_diff_ratio = diff.max()
            scores = recent_comments['body'].append(pd.Series([body])).apply(lambda x: TextBlob(x).sentiment.polarity)
            recent_avg_sentiment_polarity = scores.mean()
            recent_min_sentiment_polarity = scores.min()
                                                              
        test = [[no_follow, author_verified, author_comment_karma, author_link_karma, over_18, is_submitter, 
                 recent_num_comments, recent_num_last_30_days, recent_avg_no_follow, recent_avg_gilded, 
                 recent_avg_responses, recent_percent_neg_score, recent_avg_score, recent_min_score, 
                 recent_avg_controversiality, recent_avg_ups, recent_avg_diff_ratio, recent_max_diff_ratio, 
                 recent_avg_sentiment_polarity, recent_min_sentiment_polarity]]
        print(test)
        
        prediction = model.predict(test)
        print(prediction)

        # Return the prediction
        if prediction == 0:
            pred_text = 'Is a normal user'
        elif prediction == 1:
            pred_text = 'Is a Bot'
        elif prediction == 2:
            pred_text = 'Is a Troll'
        else:
            pred_text = 'Classification error'

        output = {'prediction': pred_text}
        return output

# Setup the Api resource routing here
# Route the URL to the resource


api.add_resource(Botidentification, '/')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
