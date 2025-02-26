#!/usr/bin/env python3
"""Importing requests"""
import requests


def sentientPlanets():
    """
    Returns list of names of all
    sentient species
    """
    url = "https://swapi-api.alx-tools.com/api/species/"
    response = requests.get(url)
    data = response.json()
    species = []
    while True:
        for result in data['results']:
            if result['designation'] == "sentient" or\
                    result['classification'] == "sentient":
                if result['homeworld']:
                    planets = requests.get(result['homeworld'])
                    name = planets.json()['name']
                    species.append(name)
        if data['next']:
            response = requests.get(data["next"])
            data = response.json()
        else:
            break
    return species
