#!/usr/bin/env python3
""" change topics"""


def update_topics(mongo_collection, name, topics):
    """
    Changes all topics of school doc
    based on name
    name(string) school name to update
    topics(list of strings) list of topics in school
    """
    mongo_collection.update_many({'name': name}, {'$set': {'topics': topics}})