#!/usr/bin/env python3

"""
contains a function that
uses Uonofficial SpaceX API to display
no. of launches per rocket

All launches should be taking in consideration
Each line should contain the rocket name and the number of
launches separated by : (colon) and space
Order the result by the number launches (descending)
If multiple rockets have the same amount of launches,
order them by alphabetic order (A to Z)
"""

import requests


def launches_per_rocket():
    """print no. of launches per rocket"""
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    data = response.json()
    rockets = {}
    for launch in data:
        rocket_id = launch['rocket']
        rocket_url = "https://api.spacexdata.com/v4/rockets/{}".format(
            rocket_id)
        # print(rocket_url)
        rocket_response = requests.get(rocket_url)
        rocket_data = rocket_response.json()
        rocket_name = rocket_data['name']
        if rocket_name in rockets:
            rockets[rocket_name] += 1
        else:
            rockets[rocket_name] = 1
    for rocket in sorted(rockets.items(), key=lambda x: (-x[1], x[0])):
        print("{}: {}".format(rocket[0], rocket[1]))


if __name__ == "__main__":
    launches_per_rocket()
