"""Create a end point for boot identification solution."""
import os
from flask import Flask, request
from flask_restful import reqparse, Api, Resource
import pickle
from model import RFModel

application = Flask(__name__)
api = Api(application)

model = RFModel()

clf_path = 'lib/models/DecisionTreeClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

clean_data_path = 'lib/models/CleanData.pkl'
with open(clean_data_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

class Botidentification(Resource):
    """Class for add end points."""

    def get(self):
        """HTTP GET /.
        """
        return {'Please send a POST request'}

    def post(self):
        """HTTP POST /.

        Boot identification solution end point.

        Returns
        -------
        Return if the JSON Data is about a Normal user, Bot or a Troll.

        """
        application.logger.info("Received request")
        
        clean_data = model.clean_data(request.data)
        application.logger.info("Cleaned data: " + str(clean_data))
        
        prediction = model.predict(clean_data)

        # Return the prediction
        if prediction == 'normal':
            pred_text = 'normal user'
        elif prediction == 'bot':
            pred_text = 'possible bot'
        elif prediction == 'troll':
            pred_text = 'possible troll'
        else:
            pred_text = 'Classification error'

        output = {'prediction': pred_text}
        application.logger.info(output)
        return output

# Setup the Api resource routing here
# Route the URL to the resource

api.add_resource(Botidentification, '/')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    application.run(debug=True, host='0.0.0.0', port=port)
