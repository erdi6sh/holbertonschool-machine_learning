#!/usr/bin/env python3
"""Retrieve starships from SWAPI that can hold a given number of passengers."""

import requests


def availableShips(passengerCount):
    """
    Return starship names that can hold at least passengerCount passengers.

    Args:
        passengerCount (int): Minimum number of passengers the starship
            must hold.

    Returns:
        list: Starship names (str) meeting the passenger requirement.
              Returns an empty list if no ships qualify.
    """
    ships = []
    url = 'https://swapi-api.hbtn.io/api/starships/'

    while url:
        res = requests.get(url)
        if res.status_code != 200:
            break

        data = res.json()
        for ship in data.get('results', []):
            passengers = ship.get('passengers', '0')
            passengers = passengers.replace(',', '').split()[0]
            try:
                num = int(passengers)
            except ValueError:
                continue
            if num >= passengerCount:
                ships.append(ship.get('name'))
        url = data.get('next')

    return ships
