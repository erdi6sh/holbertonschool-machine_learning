#!/usr/bin/env python3
"""Script to display the first upcoming SpaceX launch with key information."""

import requests
from datetime import datetime


def get_first_launch():
    """Return formatted information about the first upcoming SpaceX launch."""
    upcoming_url = 'https://api.spacexdata.com/v4/launches/upcoming'
    rockets_url = 'https://api.spacexdata.com/v4/rockets'
    launchpads_url = 'https://api.spacexdata.com/v4/launchpads'

    launches = requests.get(upcoming_url).json()
    if not launches:
        return None

    launches = [
        launch for launch in launches
        if launch.get('date_unix') is not None
    ]
    launches.sort(key=lambda launch: launch['date_unix'])
    first = launches[0]

    launch_name = first.get('name', 'Unknown')
    date_utc = first.get('date_utc')

    dt_utc = datetime.fromisoformat(date_utc.replace('Z', '+00:00'))
    dt_local = dt_utc.astimezone()
    date_local_str = dt_local.isoformat()

    rocket_id = first.get('rocket')
    launchpad_id = first.get('launchpad')

    rocket_name = 'Unknown'
    launchpad_name = 'Unknown'
    launchpad_locality = 'Unknown'

    if rocket_id:
        rocket = requests.get(f'{rockets_url}/{rocket_id}').json()
        rocket_name = rocket.get('name', 'Unknown')

    if launchpad_id:
        launchpad = requests.get(
            f'{launchpads_url}/{launchpad_id}'
        ).json()
        launchpad_name = launchpad.get('name', 'Unknown')
        launchpad_locality = launchpad.get('locality', 'Unknown')

    return (f"{launch_name} ({date_local_str}) {rocket_name} - "
            f"{launchpad_name} ({launchpad_locality})")


if __name__ == '__main__':
    info = get_first_launch()
    if info:
        print(info)
