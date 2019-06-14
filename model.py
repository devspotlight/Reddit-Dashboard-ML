"""Build Randon Forest Model."""
from sklearn.tree import DecisionTreeClassifier
import pickle
import pandas as pd
import collections
import datetime as dt
import numpy as np
from textblob import TextBlob
import json
import difflib


class RFModel(object):

    def clean_data(self, my_data):
        """Clean a dataframe.

        Parameters
        ----------
        my_data -- a JSON object from a POST request

        Returns
        -------
        Clean feature data ready to be used by the ML model

        """
        
        my_data = my_data.decode("utf-8") 
        my_data = my_data.replace('\\\\','\\')
        #print(my_data)
        
        my_data = json.loads(my_data)
        
        no_follow = bool(my_data['no_follow'])
        created_utc = int(my_data['created_utc'])
        author_verified = bool(my_data['author_verified'])
        author_comment_karma = float(my_data['author_comment_karma'])
        author_link_karma = float(my_data['author_link_karma'])
        over_18 = bool(my_data['over_18'])
        is_submitter = bool(my_data['is_submitter'])
        body = str(my_data['body'])
        recent_comments = str(my_data['recent_comments'])
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
        
        # remove escape characters to make parsing easier
        recent_comments = recent_comments.replace('\\','');
        
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
            
        return [[no_follow, author_verified, author_comment_karma, author_link_karma, over_18, is_submitter, 
                 recent_num_comments, recent_num_last_30_days, recent_avg_no_follow, recent_avg_gilded, 
                 recent_avg_responses, recent_percent_neg_score, recent_avg_score, recent_min_score, 
                 recent_avg_controversiality, recent_avg_ups, recent_avg_diff_ratio, recent_max_diff_ratio, 
                 recent_avg_sentiment_polarity, recent_min_sentiment_polarity]]

    def create(self, max_depth):
        """Create a decision tree.

        Parameters
        ----------
        max_depth -- Maximum depth of the decision tree.

        Returns
        -------
        Decision tree model.

        """
        self.clf = DecisionTreeClassifier(max_depth=3, 
                                          class_weight={'normal':1, 'bot':2.5, 'troll':5}, 
                                          min_samples_leaf=100)

    def train(self, X, y):
        """Train a model.

        Parameters
        ----------
        X -- examples
        y -- targets

        Returns
        -------
        Train Model.

        """
        self.clf.fit(X, y)

    def predict(self, X):
        """Clean a dataframe.

        Parameters
        ----------
        X -- examples

        Returns
        -------
        A prediction for the examples.

        """
        y_pred = self.clf.predict(X)
        return y_pred

    def feature_importances(self):
        """Get importance features.

        Parameters
        ----------

        Returns
        -------
        Order importance list.

        """
        return self.clf.feature_importances_

    def pickle_clf(self, path='lib/models/DecisionTreeClassifier.pkl'):
        """Saves the trained classifier for future use.

        Parameters
        ----------
        path -- folder for save the .pkl file

        Returns
        -------
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)

        print("Pickled classifier at {}".format(path))

    def pickle_clean_data(self, path='lib/models/CleanData.pkl'):
        """Saves the clean data function for future use.

        Parameters
        ----------
        path -- folder for save the .pkl file

        Returns
        -------
        """
        with open(path, 'wb') as f:
            pickle.dump(self.clean_data, f)
        print("Pickled vectorizer at {}".format(path))
