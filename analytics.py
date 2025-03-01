# this analytics API is a CLI tool that allows users to interact with the analytics database
# and perform various operations on the data. The API is designed to be user-friendly and
# easy to use. The API allows users to perform the following operations, as per `problem.md`:
# 0. Generate synthetic data, to simulate a web server log.
#   - the goal is to tell a story, and then predict that story.
#   - the story is that a user visits the site, views a page, and then books a demo, with varying probabilities for the attributes.
#   - the prediction is the probability that a user will book a demo, given the attributes, and find the important attributes.
#   - the synthetic data will be generated in a JSON file, with one JSON line per event.
#   - the synthetic data will be loaded into the database, and then the prediction will be made.
#   - the prediction will be made using a single decision tree for easy interpretation.
# 1. Event writes
#   - this will load logged JSON lines from a file into the database, as if from a web server log
#   - track_view(user_id, attributes)
#   - track_book_demo(user_id)
#   - attributes include:
#     1. on_mobile_device
#     2. in_usa
#     3. in_europe
#     4. user_agent_safari (can afford the fancy stuff)
#     5. from_facebook
#     6. from_google
#     7. from_paid
#     8. language_en
#     9. language_es
#     10. language_fr
#     11. (normally you would have something like returning visitor, but these are defined to be unchanging attributes) page_load_under_1s
#     12. accepted_all_cookies
#     13. estimated_age_over_30
#     14. bounced_on_first_visit
#     15. visit_outside_business_hours
# 2. Event reads
#   - get views of the last 24 hours
#   - get demos of the last 24 hours
# 3. Analytics
#   - moving average views(duration): the average number of views in the last duration for a six hour window every hour
#   - moving average views by attribute(duration, attribute, value): filtered by boolean attribute value
# 4. Prediction
#   - find_important_attributes(attribute_count): find the N most important attributes for booking a demo,
#                                                 and the probability of booking a demo given those attributes in a list.

import argparse
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import List, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# global variables
DB_FILE = 'analytics.db'
TABLE_NAME = 'events'
ATTRIBUTES = ['on_mobile_device', 'in_usa', 'in_europe', 'user_agent_safari', 'from_facebook', 'from_google',
              'from_paid', 'language_en', 'language_es', 'language_fr', 'page_load_under_1s', 'accepted_all_cookies',
              'estimated_age_over_30', 'bounced_on_first_visit', 'visit_outside_business_hours']
LABEL = 'book_demo'
PREDICTION = 'prediction'
PROBABILITIES = {
    "on_mobile_device": 0.55,
    "in_usa": 0.9,
    "in_europe": 0.55,
    "user_agent_safari": 0.95,
    "from_facebook": 0.6,
    "from_google": 0.6,
    "from_paid": 0.55,
    "language_en": 0.55,
    "language_es": 0.98,
    "language_fr": 0.5,
    "page_load_under_1s": 0.65,
    "accepted_all_cookies": 0.55,
    "estimated_age_over_30": 0.85,
    "bounced_on_first_visit": 0.5,
    "visit_outside_business_hours": 0.005
} # perhaps the story to tell is that the product is compelling to Spanish speakers in Safari in the USA during working hours
DB_TIME_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'

# create the database
def create_db() -> None:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(f'CREATE TABLE IF NOT EXISTS {TABLE_NAME} (timestamp TEXT, user_id TEXT, attributes JSON, event_type TEXT)')
    c.execute(f'CREATE INDEX IF NOT EXISTS idx_timestamp ON {TABLE_NAME} (timestamp)')
    # create a partial index on the event type == 'book_demo' for faster reads
    c.execute(f'CREATE INDEX IF NOT EXISTS idx_book_demo ON {TABLE_NAME} (timestamp) WHERE event_type = "book_demo"')
    conn.commit()
    conn.close()

def generate_data(file: str) -> None:
    # generate synthetic data
    # for each truth value of each boolean attribute in the probabilities dictionary, for a hundred times each,
    # generate a random user_id and a random timestamp, and write the event to the file
    # the probability that the user booked a demo is the product of the probabilities of the attributes
    with open(file, 'w') as f:
        for hours in tqdm(range(4)):
            for on_mobile_device in [True, False]:
                for in_usa in [True, False]:
                    for in_europe in [True, False]:
                        for user_agent_safari in [True, False]:
                            for from_facebook in [True, False]:
                                for from_google in [True, False]:
                                    for from_paid in [True, False]:
                                        for language_en in [True, False]:
                                            for language_es in [True, False]:
                                                for language_fr in [True, False]:
                                                    for page_load_under_1s in [True, False]:
                                                        for accepted_all_cookies in [True, False]:
                                                            for estimated_age_over_30 in [True, False]:
                                                                for bounced_on_first_visit in [True, False]:
                                                                    for visit_outside_business_hours in [True, False]:
                                                                        # user_id is a random uuid
                                                                        user_id = os.urandom(16).hex()
                                                                        # timestamp is the current date time in gmt 0 in iso format with time zone
                                                                        timestamp = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                                                                        attributes = {
                                                                            'on_mobile_device': on_mobile_device,
                                                                            'in_usa': in_usa,
                                                                            'in_europe': in_europe,
                                                                            'user_agent_safari': user_agent_safari,
                                                                            'from_facebook': from_facebook,
                                                                            'from_google': from_google,
                                                                            'from_paid': from_paid,
                                                                            'language_en': language_en,
                                                                            'language_es': language_es,
                                                                            'language_fr': language_fr,
                                                                            'page_load_under_1s': page_load_under_1s,
                                                                            'accepted_all_cookies': accepted_all_cookies,
                                                                            'estimated_age_over_30': estimated_age_over_30,
                                                                            'bounced_on_first_visit': bounced_on_first_visit,
                                                                            'visit_outside_business_hours': visit_outside_business_hours
                                                                        }
                                                                        # print([
                                                                        #     (PROBABILITIES[attr] if attributes[attr] else (1.0-PROBABILITIES[attr])) for attr in attributes.keys()
                                                                        # ])
                                                                        book_demo = (np.random.random() / 20000) < np.prod([
                                                                            (PROBABILITIES[attr] if attributes[attr] else (1.0-PROBABILITIES[attr])) for attr in attributes.keys()
                                                                        ]) # raise 20000 to raise the probability of booking a demo
                                                                        event = {
                                                                            'timestamp': timestamp,
                                                                            'user_id': user_id,
                                                                            'attributes': attributes,
                                                                            'action': 'view',
                                                                        }
                                                                        demo_booking = {
                                                                            'timestamp': timestamp,
                                                                            'user_id': user_id,
                                                                            'action': 'book_demo',
                                                                        }
                                                                        f.write(json.dumps(event) + '\n')
                                                                        if book_demo:
                                                                            f.write(json.dumps(demo_booking) + '\n')


def write_events(file: str) -> None:
    # write events to the database
    # create the database
    create_db()
    # open the file and read the events
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    with open(file, 'r') as f:
        for line in f:
            event = json.loads(line)
            c.execute(f'INSERT INTO {TABLE_NAME} (timestamp, user_id, attributes, event_type) VALUES (?, ?, ?, ?)',
                      (event['timestamp'], event['user_id'], json.dumps(event['attributes']) if event['action'] == 'view' else None, event['action']))
    conn.commit()
    conn.close()

def read_events() -> dict:
    # read events from the database
    # return a json blob of the number of views in the last 24 hours, and the number of demos in the last 24 hours
    # note that the time in the database is formatted as a string with a time zone, so we can just compare the strings
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    now = datetime.now()
    yesterday = now - timedelta(days=1)
    c.execute(f'SELECT COUNT(*) FROM {TABLE_NAME} WHERE timestamp >= ? AND timestamp <= ? AND event_type = ?',
                (yesterday.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), now.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'view'))
    views = c.fetchone()[0]
    c.execute(f'SELECT COUNT(*) FROM {TABLE_NAME} WHERE timestamp >= ? AND timestamp <= ? AND event_type = ?',
                (yesterday.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), now.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'book_demo'))
    demos = c.fetchone()[0]
    conn.close()
    print({'views_last_24_hours': views, 'demos_last_24_hours': demos})

def moving_average_views(duration: int, attribute: str = None, value: str = None) -> List[Tuple[str, float]]:
    # moving average views
    # return a list of tuples of the form (timestamp, moving average views)
    # the moving average is the average number of views in the last duration for a six hour window every hour
    # the duration can be well over a day
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    now = datetime.now()
    # check that there are no views in the future, for later debugging
    future_rows = c.execute(f'SELECT COUNT(*) FROM {TABLE_NAME} WHERE timestamp > ?', (now.strftime(DB_TIME_FORMAT),)).fetchone()[0]
    
    moving_average_window = 6
    if attribute is not None and value is not None:
        query = f'SELECT strftime("%Y-%m-%dT%H:00:00.000Z", timestamp) AS hour, COUNT(*) FROM {TABLE_NAME} WHERE timestamp >= ? AND timestamp <= ? AND event_type = ? AND json_extract(attributes, "$.{attribute}") = ? GROUP BY hour'
        params = ((now - timedelta(hours=duration + moving_average_window)).strftime(DB_TIME_FORMAT), now.strftime(DB_TIME_FORMAT), 'view', value)
    else:
        query = f'SELECT strftime("%Y-%m-%dT%H:00:00.000Z", timestamp) AS hour, COUNT(*) FROM {TABLE_NAME} WHERE timestamp >= ? AND timestamp <= ? AND event_type = ? GROUP BY hour'
        params = ((now - timedelta(hours=duration)).strftime(DB_TIME_FORMAT), now.strftime(DB_TIME_FORMAT), 'view')

    views = c.execute(query, params).fetchall()
    # serialize to json
    views = [(view[0], view[1]) for view in views]
    conn.close()
    # create the final data, which is a line for each of the hours, even if there are no views
    # this is to make the data easier to plot
    views = {view[0]: view[1] for view in views}
    moving_average_views = {}
    for i in range(duration + moving_average_window):
        hour = (now - timedelta(hours=i)).strftime('%Y-%m-%dT%H:00:00.000Z')
        if hour not in views:
            views[hour] = 0
    for i in range(duration):
        # the business value is unclear of the moving window oriented towards the future, instead of centered around the present
        # but this is changeable based on analyst feedback
        hour = (now - timedelta(hours=i)).strftime('%Y-%m-%dT%H:00:00.000Z')
        moving_average_views[hour] = sum([views[(now - timedelta(hours=i+j)).strftime('%Y-%m-%dT%H:00:00.000Z')] for j in range(moving_average_window)]) / moving_average_window
    sorted_views = sorted(moving_average_views.items(), key=lambda x: x[0])
    print(json.dumps({"views": sorted_views, "future_row_count": future_rows}))

def find_important_attributes(attribute_count: int) -> Tuple[List[str], float]:
    # find important attributes
    # return a list of the N most important attributes for booking a demo, and the probability of booking a demo given those attributes in a list
    # the prediction will be made using a single decision tree for easy interpretation
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # get the data: left join the views and the demos on the user_id, to know which views resulted in a demo
    data = c.execute(f'SELECT v.attributes, d.event_type is not null as has_booked FROM {TABLE_NAME} v LEFT JOIN {TABLE_NAME} d ON v.user_id = d.user_id and d.event_type = "book_demo" where v.event_type = "view"').fetchall()
    conn.close()
    # create the features
    # for each row in data, parse the attributes and add them to the sqlite row
    for i in (range(len(data))):
        json_data = json.loads(data[i][0])
        data[i] = [json_data[attr] for attr in ATTRIBUTES] + [data[i][1]]

    # create the dataframe
    df = pd.DataFrame(data, columns=ATTRIBUTES + ['has_booked'])
    # create the label encoder
    le = LabelEncoder()
    # encode the labels
    df['has_booked'] = le.fit_transform(df['has_booked'])
    # create the decision tree
    clf = tree.DecisionTreeClassifier(max_depth=5, random_state=42, criterion='entropy')
    # split the data, and oversample the minority class of has_booked = 1
    minority = df[df['has_booked'] == 1]
    majority = df[df['has_booked'] == 0]
    # duplicate each row in the minority class len(majority)/len(minority) times
    if len(minority) == 0:
        print("It's anyone's guess what the most important value is! We need more people to book demos. There are none")
        return
    if len(minority) > len(majority):
        minority_oversampled = minority
    else:
        minority_oversampled = minority.loc[minority.index.repeat(len(majority) // len(minority))].reset_index(drop=True)

    majority_X_train, majority_X_test, majority_y_train, majority_y_test = train_test_split(majority[ATTRIBUTES], majority['has_booked'], test_size=0.2, random_state=42)
    minority_X_train, minority_X_test, minority_y_train, minority_y_test = train_test_split(minority_oversampled[ATTRIBUTES], minority_oversampled['has_booked'], test_size=0.2, shuffle=False) # no shuffling to avoid training on test data
    X_train = pd.concat([majority_X_train, minority_X_train])
    y_train = pd.concat([majority_y_train, minority_y_train])
    X_test = pd.concat([majority_X_test, minority_X_test])
    y_test = pd.concat([majority_y_test, minority_y_test])

    # print(X_train, y_train)
    # fit the model
    clf.fit(X_train, y_train)
    # plot
    plt.figure(figsize=(40, 10))
    tree.plot_tree(clf, filled=True, feature_names=ATTRIBUTES, class_names=['no', 'yes'])
    # save the plot as a png
    plt.savefig('demo_predictor.pdf')
    # analyze the dataframe: for all of the attributes that can be true or false, what is the probability that a user with that truth value for that attribute books a demo?
    # the probability is the number of users that booked a demo with that attribute value divided by the number of users with that attribute value
    important_attributes = []
    for attr in ATTRIBUTES:
        for value in [0, 1]:
            average = df[df[attr] == value]['has_booked'].mean()
            verity = True if value == 1 else False
            important_attributes.append((attr, verity, average))

    important_attributes = sorted(important_attributes, key=lambda x: x[-1], reverse=True)
    if attribute_count is None:
        print(important_attributes[0])
    else:
        print(important_attributes[:attribute_count])

def main():
    # create the database
    create_db()
    # parse the arguments
    parser = argparse.ArgumentParser(description='Analytics API')
    permitted_commands = ['generate', 'write', 'read', 'averages', 'prediction']
    parser.add_argument('command', type=str, help='the command to run', choices=permitted_commands)
    parser.add_argument('--file', type=str, help='the file to read from or write to')
    parser.add_argument('--duration', type=int, help='the duration to use for the moving average')
    parser.add_argument('--attribute', type=str, help='the attribute to use for the moving average')
    parser.add_argument('--value', type=str, help='the value to use for the moving average')
    parser.add_argument('--attribute_count', type=int, help='the number of attributes to find')
    args = parser.parse_args()
    # run the command
    if args.command == 'generate':
        # generate synthetic data
        generate_data(args.file)
    elif args.command == 'write':
        # write events to the database
        write_events(args.file)
    elif args.command == 'read':
        # read events from the database
        read_events()
    elif args.command == 'averages':
        # perform analytics on the data
        assert args.duration is not None, 'Error: duration must be specified'
        if args.attribute is not None:
            assert args.attribute in ATTRIBUTES, f"Error: invalid attribute. Must be one of {ATTRIBUTES}"
        if args.attribute is not None and args.value is not None:
            # normalize the attribute value to sqlite's boolean value, based on whether it starts with a t or an f
            normalized_value = 1 if args.value.lower().startswith('t') else 0
            moving_average_views(args.duration, args.attribute, normalized_value)
        else:
            if args.attribute is not None:
                # this is an error, value must be specified
                print('Error: value must be specified')
            else:
                moving_average_views(args.duration)
    elif args.command == 'prediction':
        # make a prediction
        find_important_attributes(args.attribute_count)
    else:
        print('Invalid command')

if __name__ == '__main__':
    main()