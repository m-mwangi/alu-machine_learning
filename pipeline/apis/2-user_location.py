#!/usr/bin/env python3

"""
This module contains a function that
uses Github API to print location of
specific users"""

import requests
import sys
import time

# get the url from string passed on the terminal

# If the user doesn’t exist, print Not found
# If the status code is 403, print Reset in X min
# where X is the number of minutes from now and the value of X-Ratelimit-Reset
# Your code should not be executed when the file is imported
# (you should use if __name__ == '__main__':)


def print_location():
    """print location of user"""
    url = sys.argv[1]
    response = requests.get(url)
    data = response.json()

    if response.status_code == 403:
        rate_limit = int(response.headers.get('X-Ratelimit-Reset'))
        current_time = int(time.time())
        diff = (rate_limit - current_time) // 60
        print("Reset in {} min".format(diff))

    elif response.status_code == 404:
        print("Not found")
    elif response.status_code == 200:
        print(data['location'])


if __name__ == "__main__":
    print_location()
