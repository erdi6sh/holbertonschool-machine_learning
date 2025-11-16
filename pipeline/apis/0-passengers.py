#!/usr/bin/env python3
"""Module to retrieve starships from SWAPI capable of holding a given number of passengers."""

import requests


def availableShips(passengerCount):
    """
    Return a list of starship names that can hold at least passengerCount passengers.

    Args:
        passengerCount (int): The minimum number of passengers the starship must hold.

    Returns:
        list: List of starship names (str) meeting the passenger requirement.
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
            passengers = ship.get('passengers', '0').replace(',', '').split()[0]
            try:
                num = int(passengers)
            except ValueError:
                continue
            if num >= passengerCount:
                ships.append(ship['name'])
        url = data.get('next')
    return ships
