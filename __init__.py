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

    def check_habit_presence(self, utterance, time, days):
        """returns true if a habit with same
        trigger_type,time and days alreasy exists,
        returns False if not"""
        old_habits = json.load(open(self.habits_file_path))

        for old_habit in old_habits:
            for i in range(len(old_habit['intents'])):
                LOG.info('utterance')
                LOG.info(old_habit['intents'][i]['last_utterance'])
                LOG.info('---------------------------------------------')
                LOG.info(utterance)
                LOG.info('---------------------------------------------')
                LOG.info('days')
                LOG.info(days)
                LOG.info('---------------------------------------------')
                LOG.info(old_habit['days'])
                if(utterance is old_habit['intents'][i]['last_utterance']
                    # Compare hours only
                    #and time.split(":")[0] != old_habit['time'].split(":")[0]
                    and int(days) is old_habit['days']):
                    return True

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
                minute = xy[:, 1].astype(float) - hour
                hour_moy = float(mean(hour))
                min_moy = float(mean(minute))

                interval_max = max(np.absolute((hour_moy * 60 + min_moy) - hour * 60 - minute))
                # Register ID, params, intents, days, hours
                my_habit_manager = HabitsManager()
                if not my_habit_manager.check_habit_presence(str(X[0, 5]),
                                                             str(hour_moy) + ":" + str(min_moy),
                                                             str(day)):
                    my_habit_manager.register_habit("time", [
                        {
                            "name": str(X[0, 3]),
                            "parameters": X[0, 4],
                            "last_utterance": str(X[0, 5])
                        }
                    ], str(int(hour_moy)) + ":" + str(int(min_moy)), int(day), str(interval_max))
                    num_clusters = num_clusters + 1

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
    LOG.info("nb nb_clusters:%d",nb_clusters)
    LOG.info("processing finished")


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
