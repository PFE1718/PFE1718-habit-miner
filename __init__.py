# Copyright 2018 Adrien CHEVRIER, Florian HEPP, Xavier HERMAND,
#                Gauthier LEONARD, Audrey LY, Elliot MAINCOURT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import random
import csv
import numpy as np
import os
import hashlib

from datetime import datetime, timedelta

from itertools import chain, combinations
from collections import defaultdict

from sklearn.cluster import DBSCAN, AffinityPropagation
from sklearn.preprocessing import StandardScaler
from random import randrange
from datetime import datetime, timedelta, time
from dateutil import parser
from statistics import mean

from adapt.intent import IntentBuilder
from mycroft.skills.core import MycroftSkill
from mycroft.skills.core import intent_handler
from mycroft.skills.context import adds_context, removes_context
from mycroft.util.log import getLogger, LOG

__author__ = 'Nuttymoon'

# Logger: used for debug lines, like "LOGGER.debug(xyz)". These
# statements will show up in the command line when running Mycroft.
LOGGER = getLogger(__name__)

class HabitsManager(object):
    """
    This class manages the reading and writting in the file habits.json

    Attributes:
        habits_file_path (str): path to the file habits.json
        habits (json): the json datastore corresponding to habits.json
    """

    def __init__(self):
        self.habits_file_path = "/opt/mycroft/habits/habits.json"
        self.triggers_file_path = "/opt/mycroft/habits/triggers.json"
        self.load_files()

    def load_files(self):
        self.habits = json.load(open(self.habits_file_path))
        self.triggers = json.load(open(self.triggers_file_path))

    def get_all_habits(self):
        """Return all the existing habits of the user"""
        return self.habits

    def get_habit_by_id(self, habit_id):
        """Return one particular habit of the user"""
        return self.habits[habit_id]

    def check_skill_habit(self, new_habit):
        """
        Verify if skill habit is already present
        """

        old_habits = self.get_all_habits()
        new_utterances=[str(utt['utterance']) for utt in new_habit]
        # Check utterances for each old habit
        for ohabit in old_habits:
            # Extract utterances from old habit
            old_utterances = [str(outt['last_utterance']) for outt in ohabit['intents']]
            # Check if new utterances are in old utterances
            if set(new_utterances).issubset(set(old_utterances)):
                LOG.info('skill habit already exists')
                return True
        LOG.info('skill habit does not exist')
        return False

    def check_habit_presence(self, utterance, time, days):
        """returns true if a habit with same
        trigger_type,time and days alreasy exists,
        returns False if not"""
        old_habits = self.get_all_habits()

        for old_habit in old_habits:
            if old_habit['trigger_type'] in "time":
                for oi in old_habit['intents']:
                    if((str(utterance) in str(oi['last_utterance'])) and (str(int(float(days))) is str(old_habit['days']))):
                        LOG.info("OLD HABIT FOUND")
                        return True
        LOG.info("NO HABIT FOUND")
        return False

    def register_habit(self, trigger_type, intents, t=None, days=None,
                       interval_max=None):
        """Register a new habit in habits.json"""
        if trigger_type == "skill":
            self.habits += [
                {
                    "intents": intents,
                    "trigger_type": trigger_type,
                    "automatized": 0,
                    "user_choice": False,
                    "triggers": []
                }
            ]
        else:
            self.habits += [
                {
                    "intents": intents,
                    "trigger_type": trigger_type,
                    "automatized": 0,
                    "user_choice": False,
                    "time": t,
                    "days": days,
                    "interval_max": interval_max
                }
            ]
        with open(self.habits_file_path, 'w') as habits_file:
            json.dump(self.habits, habits_file)

    def automate_habit(self, habit_id, auto, new_triggers=None):
        """
        Register the automation of a habit in the habits.json

        Args:
            habit_id (str): the id of the habit to automate
            triggers (str[]): the intents to register as triggers of the habit
            auto (int): 1 for full automation, 2 for habit offer when triggered
        """
        habit = self.habits[habit_id]
        habit["user_choice"] = True
        habit["automatized"] = auto

        if habit["trigger_type"] == "skill":
            if not self.triggers:
                for i in new_triggers:
                    self.triggers += [
                        {
                            "intent": habit["intents"][i]["name"],
                            "parameters": habit["intents"][i]["parameters"],
                            "habit_id": habit_id
                        }
                    ]
            else:
                if not self.check_triggers(habit_id, habit, new_triggers):
                    return False

            habit["triggers"] = new_triggers
            with open(self.triggers_file_path, 'w+') as triggers_file:
                json.dump(self.triggers, triggers_file)

        self.habits[habit_id] = habit
        with open(self.habits_file_path, 'w+') as habits_file:
            json.dump(self.habits, habits_file)

        return True

    def check_triggers(self, habit_id, habit, new_triggers):
        to_add = []
        for known_trig in self.triggers:
            for i in new_triggers:
                LOGGER.info("Testing trigger" + str(habit["intents"][int(i)]))
                if habit["intents"][i]["name"] == known_trig["intent"] and \
                        habit["intents"][i]["parameters"] \
                        == known_trig["parameters"]:
                    return False
                to_add += [
                    {
                        "intent": habit["intents"][i]["name"],
                        "parameters": habit["intents"][i]["parameters"],
                        "habit_id": habit_id
                    }
                ]
        self.triggers += to_add

        return True

    def not_automate_habit(self, habit_id):
        """
        Register the user choice of not automatizing a habit

        Args:
            habit_id (str): the id of the habit to not automate
        """
        self.habits[habit_id]["user_choice"] = True
        self.habits[habit_id]["automatized"] = 0
        with open(self.habits_file_path, 'w+') as habits_file:
            json.dump(self.habits, habits_file)

    def get_trigger_by_id(self, trigger_id):
        """Return one particular habit trigger"""
        return self.triggers[trigger_id]


class HabitMinerSkill(MycroftSkill):
    """
    This class implements the habit miner skill

    Attributes:

    """

    def __init__(self):
        super(HabitMinerSkill, self).__init__(
            name="HabitMinerSkill")
        self.logs_file_path = "/opt/mycroft/habits/logs.json"

    @intent_handler(IntentBuilder("LaunchMiningIntent")
                    .require("LaunchMiningKeyword"))
    def handle_launch_mining(self, message):
        process_mining(self.logs_file_path)

    def stop(self):
        pass


def create_skill():
    return HabitMinerSkill()


# Read json logs


def read_json(logs_file_path):
    data = []
    # Open Json
    for line in open(logs_file_path, 'r+'):
        data.append(json.loads(line))
    return data


# Parse json and extract variables variables


def parse_json(data):
    days = []
    times = []
    ids = []
    intents = []
    params = []
    utt = []
    # parse each json variable
    for i in range(len(data)):
        datetime_object = parser.parse(data[i]["datetime"])
        # Extract day of week
        my_day = date_to_days(datetime_object)
        days.append(my_day)
        # Extract hour of day
        my_hour = time_to_hours(datetime_object)
        times.append(my_hour)
        # Extract intents
        my_id = str(data[i]["type"]) + (str(data[i]["parameters"]))
        m_param = str(data[i]["parameters"])
        ids.append(my_id)
        my_intent = str(data[i]["type"])
        intents.append(my_intent)
        my_param = data[i]["parameters"]
        params.append(my_param)
        my_utterance = str(data[i]["utterance"])
        utt.append(my_utterance)
    X = np.array((days, times, ids, intents, params, utt))
    X = X.transpose()
    print(X.shape)
    return X


def extract_ids(data):
    unique_ids = np.unique(data)
    return unique_ids


def random_date(start, end):
    """
    This function will return a random datetime between two datetime
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)


def date_to_days(date):
    """
    This function returns day of week in range 1-10
    """
    day_t = (float(date.weekday()) *10.0)/6.0
    return day_t


def time_to_seconds(date):
    """
    Function returns time of the day in seconds"""
    td = date.time()
    return ((td.hour * 3600 + td.second) * 10 ** 6 + td.microsecond) / 10 ** 6


def time_to_hours(date):
    """
    Function returns time of the day in hours in range 1-10"""
    td = date.time()
    td_t = (((float(td.hour) * 60 + float(td.minute)) / 60) * 10.0) / 24.0
    return float(td_t)


# Write new habit
def write_habit(X, labels,core_samples_mask_dbscan):
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    num_clusters = 0

    for k in unique_labels:
        #Check if cluster
        if k >=0:

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask_dbscan]
            if (len(xy[:, 1]) > 1):
                day = round(mean(xy[:, 0].astype(float)), 0)
                hour = xy[:, 1].astype(float).astype(int)
                minute = (xy[:, 1].astype(float) - hour)
                minute = map(lambda x: x * 60, minute)
                # calculate mean hour and minute of the cluster
                hour_moy = float(mean(hour))
                min_moy = int(float(mean(minute)))
                # Round time to 5 minutes
                min_moy_rounded = float(float(min_moy) / 60.0)
                min_moy_rounded = int(round(min_moy_rounded * 12) * 5)
                # Calculate inteval max to detect habit
                interval_max = np.ceil(max(np.absolute((hour_moy * 60 + min_moy) - hour * 60 - minute)))
                # Register ID, params, intents, days, hours
                my_habit_manager = HabitsManager()
                if not my_habit_manager.check_habit_presence(str(X[0, 5]),
                                                             str(hour_moy) + ":" + str(min_moy_rounded),
                                                             str(day)):
                    LOG.info("NEW CLUSTER WRITTEN")
                    my_habit_manager.register_habit("time", [
                        {
                            "name": str(X[0, 3]),
                            "parameters": X[0, 4],
                            "last_utterance": str(X[0, 5])
                        }
                    ], str(int(hour_moy)) + ":" + str(int(min_moy_rounded)), int(day), str(interval_max))
                    num_clusters = num_clusters + 1
                else:
                    LOG.info("NO CLUSTER WRITTEN")


    return num_clusters


# MAIN STEPS
# Learning steps


def process_mining(logs_file_path):
    # READ DATAimport sys

    try:
        raw_data = read_json(logs_file_path)
        X = parse_json(raw_data)
        # Take hour and day for clustering
        X_cluster = X[:, [0, 1]]
    except:
        print "Unexpected error, exiting habit miner"
        return -1


    # compute dbscan per id
    unique_ids = extract_ids(X[:,2])
    nb_clusters = 0

    for i in unique_ids:
        # even_numbers = list(filter(lambda x: x % 2 == 0, fibonacci))
        X_mini = X[X[:, 2] == i]
        if (X_mini[:, 0].shape[0] > 2):
            X_mini_cluster = X_mini[:, [0, 1]]
            # Affinity propagation
            # cluster_centers_indices, labels = compute_AP(X_mini_cluster)
            # dbscan
            core_samples_mask, labels = compute_DBSCAN(X_mini_cluster)

            # convert days and hours to 24 and 7 ranges

            def to_24(x): return (x / 10) * 7

            def to_7(x): return (x / 10) * 24

            X_mini[:, 0] = to_24(X_mini[:, 0].astype(float))
            X_mini[:, 1] = to_7(X_mini[:, 1].astype(float))
            X_mini_cluster[:, 0] = to_24(X_mini_cluster[:, 0].astype(float))
            X_mini_cluster[:, 1] = to_7(X_mini_cluster[:, 1].astype(float))

            nb_clusters = write_habit(X_mini, labels,core_samples_mask) + nb_clusters
            # Plot AP
            # plot_AP(X_mini_cluster,cluster_centers_indices,labels)
            # plot dbscan
            name = X_mini[0, 2]
            #plot_DBSCAN(X_mini_cluster, core_samples_mask, labels, name)
    LOG.info("nb nb_clusters:%d",nb_clusters)
    LOG.info("processing finished")

    run_apriori(logs_file_path)
    LOG.info("apriori finished")


# MODELS
# Compute Affinity Propagation
def compute_AP(features):
    af = AffinityPropagation(preference=-50, damping=0.9).fit(features)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    return cluster_centers_indices, labels


def compute_DBSCAN(features):
    features = StandardScaler().fit_transform(features.astype(float))
    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=0.25, min_samples=4).fit(features)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    return core_samples_mask, labels


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def return_item_with_min_support(item_set, transaction_list, min_support, freq_set):
    """calculates the support for items in the itemSet and returns a subset
   of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    local_set = defaultdict(int)

    for item in item_set:
        for transaction in transaction_list:
            if item.issubset(transaction):
                freq_set[item] += 1
                local_set[item] += 1

    for item, count in local_set.items():
        support = float(count) / len(transaction_list)

        if support >= min_support:
            _itemSet.add(item)

    return _itemSet


def join_set(item_set, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set([i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length])


def get_item_set_transaction_list(data_iterator):
    transaction_list = list()
    item_set = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transaction_list.append(transaction)
        for item in transaction:
            item_set.add(frozenset([item]))  # Generate 1-itemSets
    return item_set, transaction_list


def apriori(data_iter, min_support, min_confidence):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    item_set, transaction_list = get_item_set_transaction_list(data_iter)

    freq_set = defaultdict(int)
    large_set = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    assocRules = dict()
    # Dictionary which stores Association Rules

    one_c_set = return_item_with_min_support(item_set,
                                           transaction_list,
                                           min_support,
                                           freq_set)

    current_l_set = one_c_set
    k = 2
    while current_l_set != set([]):
        large_set[k - 1] = current_l_set
        current_l_set = join_set(current_l_set, k)
        current_c_set = return_item_with_min_support(current_l_set,
                                                   transaction_list,
                                                   min_support,
                                                   freq_set)
        current_l_set = current_c_set
        k = k + 1

    def get_support(item):
        """local function which Returns the support of an item"""
        return float(freq_set[item]) / len(transaction_list)

    to_ret_items = []
    for key, value in large_set.items():
        to_ret_items.extend([(tuple(item), get_support(item))
                           for item in value])

    to_ret_rules = []
    for key, value in large_set.items()[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = get_support(item) / get_support(element)
                    if confidence >= min_confidence:
                        to_ret_rules.append(((tuple(element), tuple(remain)), confidence))
    return to_ret_items, to_ret_rules


def print_results(items, rules):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    for item, support in sorted(items, key=lambda (item, support): support):
        print "item: %s , %.3f" % (str(item), support)
    print "\n------------------------ RULES:"
    for rule, confidence in sorted(rules, key=lambda (rule, confidence): confidence):
        pre, post = rule
        print "Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence)


def data_from_file(fname):
    """Function which reads from the file and yields a generator"""
    file_iter = open(fname, 'rU')
    for line in file_iter:
        line = line.strip().rstrip(',')  # Remove trailing comma
        record = frozenset(line.split(','))
        yield record


def run_apriori(logs_file_path, min_supp=0.05, min_confidence=0.8):
    hashes_temp = []
    table_csv = []
    date_time_obj0 = datetime.strptime('2018-01-01 00:00:00.0', '%Y-%m-%d %H:%M:%S.%f')
    habit_manager = HabitsManager()

    # Open logs and put them in a list, same line if consequent logs are within 5 minutes interval
    with open(logs_file_path) as json_data:
        for i, line in enumerate(json_data):
            data = json.loads(line)
            date_time_obj1 = datetime.strptime(data['datetime'], '%Y-%m-%d %H:%M:%S.%f')
            delta = date_time_obj1 - date_time_obj0
            del data['datetime']
            hash = hashlib.md5(json.dumps(data)).hexdigest()

            if delta > timedelta(minutes=5):
                table_csv.append(hashes_temp)
                hashes_temp = []
                hashes_temp.append(hash)
                date_time_obj0 = date_time_obj1
            else:
                hashes_temp.append(hash)

    del table_csv[0]

    # Converts logs list to csv so that we can executre apriori algorithm on them
    with open('/opt/mycroft/habits/inputApriori.csv', 'w') as fp:
        LOG.info('opened')
        writer = csv.writer(fp, delimiter=',')
        for row in table_csv:
            writer.writerow(row)

    in_file = data_from_file('/opt/mycroft/habits/inputApriori.csv')

    items, rules = apriori(in_file, min_supp, min_confidence)

    # reformat the rules and sort the tuples in it
    hashes_temp = []

    for rule in rules:
        hash = rule[0][0] + rule[0][1]
        hashes_temp.append(sorted(hash))

    # this is to remove duplicates
    hashes = []

    for tuple in hashes_temp:
        if tuple not in hashes:
            hashes.append(tuple)

    # Hash to json data
    habits = []
    habit = []
    intents = []

    for hash in hashes:
        for intent in hash:
            with open(logs_file_path) as json_data:
                for line in json_data:
                    data = json.loads(line)
                    del data['datetime']
                    hash = hashlib.md5(json.dumps(data)).hexdigest()
                    if hash == intent:
                        habit.append(json.dumps(data))
                        intents.append(data)
                        break
        if not habit_manager.check_skill_habit(intents):
            habits.append(habit)
        habit = []

    # format habits as expected and register them
    intents = []

    for habit in habits:
        for intent in habit:
            intent = {
                'last_utterance': json.loads(intent)['utterance'],
                'name': json.loads(intent)['type'],
                'parameters': json.loads(intent)['parameters']
            }
            intents.append(intent)
        habit_manager.register_habit("skill", intents)
        intents = []

    os.remove('/opt/mycroft/habits/inputApriori.csv')



