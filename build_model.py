"""Build a model for Random Forest."""
from model import RFModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from IPython import get_ipython
import seaborn as sns
import numpy as np
import collections
from datetime import datetime


def build_model():
    """Create a .pkl file for Random Forest from comments.csv."""
    model = RFModel()

    with open('lib/data/comments.csv') as f:
        my_data = pd.read_csv(f, sep=',')

    # Clean dataset
    old_data = my_data.copy()
    my_data = model.clean_data(my_data)

    # Delete irrelevant columns
    columns = ['approved_at_utc', 'mod_reason_by', 'banned_by',
               'removal_reason', 'likes', 'banned_at_utc', 'mod_reason_title',
               'downs', 'num_reports', 'quarantine']
    my_data.drop(columns, inplace=True, axis=1)
    print("Original size of dataset: ", my_data.shape)
    print("After removing empty columns: ", old_data.shape)

    # Kepp commets for entropy
    my_data['body'] = old_data['body']
    my_data['link_title'] = old_data['link_title']

    # Count num of bots a trolls
    bots = my_data['is_bot'].values
    trolls = my_data['is_troll'].values
    print("Number of bots: ", bots.sum())
    print("Number of trolls:", trolls.sum())

    # Change 1 to 2 for a troll
    trolls = trolls * 2

    # Create a one columns for troll and  bots
    targets = bots + trolls

    # Delete is_bot and is_troll collumns and add targets column
    columns = ['is_bot', 'is_troll']
    my_data.drop(columns, inplace=True, axis=1)
    my_data['target'] = targets

    # Num of users
    users = my_data['author'].values
    num_of_users = np.unique(users)
    print("Number of author or users: ", len(num_of_users))
    my_data.to_csv('lib/data/my_clean_data.csv', sep=',', index=False)
    print("The data cleaning finished correctly!!!")
    print("Clean Dataset Shape: ", my_data.shape)
    print("Clean Dataset Columns: ", my_data.columns)
    
    # Convert created utc with datatime and detect negative score
    times = my_data['created_utc'].tolist()
    negative_score = my_data['score'].tolist()
    for idx in range(len(times)):
        times[idx] = datetime.utcfromtimestamp(times[idx])
        negative_score[idx] = 1 if negative_score[idx] < 0 else 0
    lastday = max(times)
    my_data['negative_score'] = negative_score
    
    # Detect comments 30 days prior to the last comment
    for idx in range(len(times)):
        date = times[idx]
        if(date.year == lastday.year and date.month == lastday.month):
            times[idx] = 1
        elif(date.year == lastday.year and (lastday.month - 1) == date.month and date.day > lastday.day ):
            times[idx] = 1
        else:
            times[idx] = 0
    my_data['created_utc'] = times

    # Delete irrelevant columns
    columns = ['id', 'author_created_at', 'subreddit_id', 'link_id',
               'author_flair_template_id', 'archived', 'body', 'link_title',
               'subreddit', 'subreddit_type']
    my_data.drop(columns, inplace=True, axis=1)
    print("After removing columns not considered: ", my_data.shape)
    print("Considered columns: ", my_data.columns)

    # Describe the new dataset
    features = my_data.describe()

    # Create new Dataset
    dataset = pd.DataFrame()
    for feature in features:
        # Calculate mean, mad and meadian of score, ups and controversiality group by user
        if(feature == 'score' or feature == 'ups' or feature == 'controversiality'):
            groupby_autor = my_data[feature].groupby(my_data['author']).median()
            dataset['median_' + feature] = groupby_autor
            groupby_autor = my_data[feature].groupby(my_data['author']).mad()
            dataset['mad_' + feature] = groupby_autor
        # Calculate mean of all features group by user
        if(feature != 'author'):
            groupby_autor = my_data[feature].groupby(my_data['author']).mean()
            if(feature != 'target'):
                dataset['mean_' + feature] = groupby_autor
            else:
                dataset[feature] = groupby_autor
        # Calculate percent of past comments with a negative score group by user        
        if(feature == 'negative_score'):
            groupby_autor = my_data[feature].groupby(my_data['author']).sum()
            count_groupby_autor = my_data[feature].groupby(my_data['author']).count()
            dataset['percent_comments_negative_score'] = groupby_autor / count_groupby_autor 
        # Calculate number of comments the user made in the 30 days prior to the last comment
        if(feature == 'created_utc'):
            groupby_autor = my_data[feature].groupby(my_data['author']).sum()
            dataset['num_comments_last30day'] = groupby_autor
        if(feature == 'target'):
            dataset['target'] = dataset['target'].round(0)

    dataset = dataset[['mean_author_link_karma', 'mean_author_comment_karma',
       'mean_author_verified', 'mean_author_has_verified_email', 'mean_edited',
       'mean_author_flair_type', 'mean_gilded', 'mean_no_follow',
       'median_score', 'mad_score', 'mean_score','mean_over_18',
       'median_controversiality','mad_controversiality',
       'mean_controversiality', 'mean_is_submitter',
       'num_comments_last30day', 'median_ups', 'mad_ups', 'mean_ups',
       'percent_comments_negative_score', 'target']]

    # Number of targets
    targets = collections.Counter(dataset['target'])
    print(targets)

    # Plot number of features
    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic('matplotlib', 'inline')
    # Creating a bar plot
    sns.set(style="darkgrid")
    sns.countplot(x="target", data=dataset)
    # Add labels to your graph
    plt.xlabel('Target Score')
    plt.ylabel('Targets')
    plt.title("Targets Distribution")
    plt.show()

    print("New Dataset Shape: ", dataset.shape)
     # Save new dateset
    dataset.to_csv('lib/data/new_data.csv', sep=',', index=False)
    # Extract feature and target np arrays (inputs for placeholders)
    data = dataset
    input_x = data.iloc[:, 0:-1].values
    input_y = data.iloc[:, -1].values

    print("Preview Features Dataset Format: ", input_x[0:3])
    print("Preview Targets Dataset Format: ", input_y[0:3])

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
            input_x, input_y, test_size=0.3, random_state=16)

    # Create a Gaussian Classifier
    model.create(100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    y_true = y_test
    matrix = pd.crosstab(y_true, y_pred, rownames=['True'],
                         colnames=['Predicted'], margins=True)
    print(matrix)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Mcc:", metrics.matthews_corrcoef(y_test, y_pred))
    print("F1 :", metrics.f1_score(y_test, y_pred, average=None))
    print("Recall :", metrics.recall_score(y_test, y_pred, average=None))
    print("Precision:", metrics.precision_score(y_test, y_pred, average=None))

    # Get feature importances
    # Create a Gaussian Classifier
    model.create(n_estimators=1000)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    model.train(X_train, y_train)

    feature_imp = pd.Series(
            model.feature_importances(),
            index=data.columns.drop('target')).sort_values(ascending=False)
    feature_imp

    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic('matplotlib', 'inline')
    # Creating a bar plot
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.show()

    # Selected Feactures and created new dataset
    newdata = data[['num_comments_last30day', 'mad_score', 'mad_ups', 'mean_ups',
                    'mean_author_verified', 'mean_score', 'percent_comments_negative_score',
                    'mean_author_link_karma','mean_controversiality', 'mean_author_comment_karma',
                    'mad_controversiality', 'mean_no_follow', 'target']]
    input_x = newdata.iloc[:, 0:-1].values
    input_y = newdata.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
            input_x, input_y, test_size=0.3, random_state=16)

    # Create a Gaussian Classifier
    model.create(1000)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    model.train(X_train, y_train)

    # prediction on test set
    y_pred = model.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    y_true = y_test

    matrix = pd.crosstab(y_true, y_pred, rownames=['True'],
                         colnames=['Predicted'], margins=True)
    print(matrix)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Mcc:", metrics.matthews_corrcoef(y_test, y_pred))
    print("F1 :", metrics.f1_score(y_test, y_pred, average=None))
    print("Recall :", metrics.recall_score(y_test, y_pred, average=None))
    print("Precision:", metrics.precision_score(y_test, y_pred, average=None))

    model.pickle_clf()
    model.pickle_clean_data()


if __name__ == "__main__":
    build_model()
