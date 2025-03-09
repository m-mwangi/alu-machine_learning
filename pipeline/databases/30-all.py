#!/usr/bin/env python3
"""
Function that lists all documents in a collection
"""


def list_all(mongo_collection):
    """
    returns empty list if no doc
    in collection
    """
    docs = []
    collection = mongo_collection.find()
    for doc in collection:
        docs.append(doc)
    return docs