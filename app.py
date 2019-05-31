"""Create a end point for boot identification solution."""
import os
from flask import Flask
from flask_restful import reqparse, Api, Resource
import pickle
from model import RFModel
import sys

app = Flask(__name__)
api = Api(app)

model = RFModel()

clf_path = 'lib/models/DecisionTreeClassifier.pkl'
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
        
        print("recent:", args['recent_comments'])
        
        clean_data = model.clean_data(args)
        print(clean_data)
        
        prediction = model.predict(clean_data)
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
