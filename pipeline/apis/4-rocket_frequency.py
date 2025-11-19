#!/usr/bin/env python3
"""Script to display the number of SpaceX launches per rocket."""

import requests


def rocket_frequencies():
    """
    Compute the number of launches per rocket.

    Returns:
        list of tuples: (rocket_name, count) sorted by:
            - count descending
            - name ascending for ties
    """
    launches_url = 'https://api.spacexdata.com/v4/launches'
    rockets_url = 'https://api.spacexdata.com/v4/rockets'

    launches = requests.get(launches_url).json()
    rockets = requests.get(rockets_url).json()

    # Map rocket id to name
    rocket_id_to_name = {
        rocket.get('id'): rocket.get('name', 'Unknown')
        for rocket in rockets
    }

    counts = {}
    for launch in launches:
        rocket_id = launch.get('rocket')
        if not rocket_id:
            continue
        name = rocket_id_to_name.get(rocket_id, 'Unknown')
        counts[name] = counts.get(name, 0) + 1

    # Sort by count desc, then name asc
    sorted_items = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0])
    )
    return sorted_items


if __name__ == '__main__':
    for name, count in rocket_frequencies():
        print(f"{name}: {count}")
