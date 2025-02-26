#!/usr/bin/env python3

"""
This script uses the GitHub API to print the location of a specific user.
"""

import requests
import sys
import time

def print_location():
    """Prints the location of a GitHub user from the API."""
    
    # Ensure a URL argument is provided
    if len(sys.argv) < 2:
        print("Usage: ./2-user_location.py <GitHub API URL>")
        sys.exit(1)

    url = sys.argv[1]
    response = requests.get(url)

    if response.status_code == 403:
        rate_limit = int(response.headers.get('X-Ratelimit-Reset', 0))
        current_time = int(time.time())
        diff = (rate_limit - current_time) // 60
        print("Reset in {} min".format(diff))
        return  # Stop execution

    elif response.status_code == 404:
        print("Not found")
        return  # Stop execution

    elif response.status_code == 200:
        data = response.json()
        print(data.get('location', "No location available"))

if __name__ == "__main__":
    print_location()
