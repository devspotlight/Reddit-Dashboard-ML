"""Build Randon Forest Model."""
from sklearn.ensemble import RandomForestClassifier
from datacleaner import autoclean
import pickle


class RFModel(object):
    """Build Randon Forest Model for save this in .pkl file."""
    def __init__(self):
        """Define function for create .pkl file."""
        self.clf = RandomForestClassifier()
        self.clean = autoclean

    def clean_data(self, my_data):
        """Clean a dataframe.

        Parameters
        ----------
        my_data -- data for clean

        Returns
        -------
        Clean data.

        """
        return self.clean(my_data)

    def create(self, n_estimators):
        """Create a Random Forest Model.

        Parameters
        ----------
        n_estimator -- parameter for Random Forest Model.

        Returns
        -------
        Randon Forest model.

        """
        self.clf = RandomForestClassifier(n_estimators, max_depth=2)

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

    def pickle_clf(self, path='lib/models/RandomForestClassifier.pkl'):
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
            pickle.dump(self.clean, f)
        print("Pickled vectorizer at {}".format(path))
