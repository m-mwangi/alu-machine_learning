#!/usr/bin/env python3
"""Function to list school"""


def schools_by_topic(mongo_collection, topic):
    """
    Returns list of school having specific topic
    """
    schools = []
    collections = mongo_collection.find({'topics': topic})
    for doc in collections:
        schools.append(doc)
    return schools