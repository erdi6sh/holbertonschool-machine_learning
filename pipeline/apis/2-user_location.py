#!/usr/bin/env python3
"""Script to print the location of a GitHub user from their API URL."""

import sys
from datetime import datetime
import requests


def print_user_location(api_url):
    """Fetch a GitHub user and print their location or rate-limit info."""
    try:
        res = requests.get(api_url)
    except requests.RequestException:
        print("Not found")
        return

    if res.status_code == 200:
        data = res.json()
        location = data.get('location')
        if location:
            print(location)
        else:
            print("Not found")
    elif res.status_code == 404:
        print("Not found")
    elif res.status_code == 403:
        reset = res.headers.get("X-RateLimit-Reset")
        try:
            reset_ts = int(reset)
            now_ts = int(datetime.now().timestamp())
            wait_sec = reset_ts - now_ts
            wait_min = max(wait_sec // 60, 0)
        except (TypeError, ValueError):
            wait_min = 0
        print(f"Reset in {wait_min} min")
    else:
        print("Not found")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(1)
    print_user_location(sys.argv[1])
