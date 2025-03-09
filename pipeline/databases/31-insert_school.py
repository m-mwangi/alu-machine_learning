#!/usr/bin/env python3
"""Function to insert new document"""


def insert_school(mongo_collection, **kwargs):
    """
    returns new _id
    """
    document = mongo_collection.insert_one(kwargs)
    return (document.inserted_id)