"""Cleaning comments dataset."""
import pandas as pd
from datacleaner import autoclean
import numpy as np


def clean_data():
    """Clean comments.csv file and saved the new values in clean_data.csv."""
    with open('lib/data/comments.csv') as f:
        my_data = pd.read_csv(f, sep=',')

    # Clean dataset
    old_data = my_data.copy()
    my_data = autoclean(my_data)

    # Delete irrelevant columns
    columns = ['approved_at_utc', 'mod_reason_by', 'banned_by',
               'removal_reason', 'likes', 'banned_at_utc', 'mod_reason_title',
               'downs', 'num_reports', 'quarantine']
    my_data.drop(columns, inplace=True, axis=1)
    print("Original size of dataset: ", old_data.shape)
    print("After removing empty columns: ", my_data.shape)

    # Kepp commets for entropy
    my_data['body'] = old_data['body']
    my_data['link_title'] = old_data['link_title']

    # Count num of bots a trolls
    bots = my_data['is_bot'].values
    trolls = my_data['is_troll'].values
    print("Number of bot comments: ", bots.sum())
    print("Number of bots: ", len(np.unique(bots)))
    print("Number of troll comments:", trolls.sum())
    print("Number of trolls:", len(np.unique(bots)))

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


if __name__ == "__main__":
    clean_data()
