"""Create a end point for boot identification solution."""
import os
from flask import Flask
from flask_restful import reqparse, Api, Resource
import pickle
from model import RFModel

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

parser.add_argument('num_comments_last30day')
parser.add_argument('mad_score')
parser.add_argument('mad_ups')
parser.add_argument('mean_ups')
parser.add_argument('mean_author_verified')
parser.add_argument('mean_score')
parser.add_argument('percent_comments_negative_score')
parser.add_argument('mean_author_link_karma')
parser.add_argument('mean_controversiality')
parser.add_argument('mean_author_comment_karma')
parser.add_argument('mad_controversiality')
parser.add_argument('mean_no_follow')


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

        Form Data. This is not necesary for demo user.
        ---------
        num_comments_last30day -- number of comments the user made in the last 30 days.

        mad_score -- Mean absolute deviation score of a user.

        mad_ups -- Mean absolute deviation ups of a user.

        mean_ups -- Average ups of a user.

        mean_author_verified -- Average author_verified of a user.

        mean_score -- Average score of a user.

        percent_comments_negative_score -- Percent of past comments with a negative score of a user.

        mean_author_link_karma_verified -- Average author_link_karma_verified of a user.

        mean_controversiality -- Average controversiality of a user.

        mean_author_comment_karma -- Average author_comment_karma of a user.

        mad_controversiality -- Mean absolute deviation controversiality of a user.

        mean_no_follow -- Average no_follow of a user.

        Returns
        -------
        Return if the Form Data is about a Normal user, Bot or a Troll.

        """
        # Use parser and find the user's query
        args = parser.parse_args()

        num_comments_last30day = float(args['num_comments_last30day'])
        mad_score = float(args['mad_score'])
        mad_ups = float(args['mad_ups'])
        mean_ups = float(args['mean_ups'])
        mean_author_verified = float(args['mean_author_verified'])
        mean_score = float(args['mean_score'])
        percent_comments_negative_score = float(args['percent_comments_negative_score'])
        mean_author_link_karma = float(args['mean_author_link_karma'])
        mean_controversiality = float(args['mean_controversiality'])
        mean_author_comment_karma = float(args['mean_author_comment_karma'])
        mad_controversiality = float(args['mad_controversiality'])
        mean_no_follow = float(args['mean_no_follow'])

        test = [[num_comments_last30day, mad_score, mad_ups, mean_ups,mean_author_verified, 
                 mean_score, percent_comments_negative_score, mean_author_link_karma,
                 mean_controversiality, mean_author_comment_karma, mad_controversiality,
                 mean_no_follow]]
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
