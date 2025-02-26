#!/usr/bin/env python3

"""
This module contains a function that
uses Uonofficial SpaceX API to display
upcoming launch with these information:
    - Name of the launch
    - Date (in local time)
    - Rocket name
    - Launchpad name
format of the output:
<launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)
"""

import requests


def upcomingLaunch():
    """print upcoming launch"""
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(url)
    data = response.json()
    recent = 0

    for dic in data:
        new = int(dic["date_unix"])
        if recent == 0 or new < recent:
            recent = new
            launch_name = dic["name"]
            date = dic["date_local"]
            rocket_number = dic["rocket"]
            launch_number = dic["launchpad"]

    # launch = data[0]
    # rocket_id = launch['rocket']
    # launchpad_id = launch['launchpad']
    rocket_url = "https://api.spacexdata.com/v4/rockets/{}".format(
        rocket_number)
    launchpad_url = "https://api.spacexdata.com/v4/launchpads/{}".format(
        launch_number)
    rocket_response = requests.get(rocket_url)
    launchpad_response = requests.get(launchpad_url)
    rocket_data = rocket_response.json()
    launchpad_data = launchpad_response.json()
    rocket_name = rocket_data['name']
    launchpad_name = launchpad_data['name']
    launchpad_locality = launchpad_data['locality']
    print("{} ({}) {} - {} ({})".format(
        launch_name,
        date,
        rocket_name,
        launchpad_name,
        launchpad_locality
    ))


if __name__ == "__main__":
    upcomingLaunch()
