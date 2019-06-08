"""Create a end point for boot identification solution."""
import os
from flask import Flask, request
from flask_restful import reqparse, Api, Resource
import pickle
from model import RFModel

app = Flask(__name__)
api = Api(app)

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
        app.logger.info("Received request")
        
        clean_data = model.clean_data(request.data)
        app.logger.info("Cleaned data: " + str(clean_data))
        
        prediction = model.predict(clean_data)

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
        app.logger.info(output)
        return output

# Setup the Api resource routing here
# Route the URL to the resource

api.add_resource(Botidentification, '/')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
