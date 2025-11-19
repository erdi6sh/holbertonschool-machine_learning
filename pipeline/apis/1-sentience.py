#!/usr/bin/env python3
"""Module to find home planets of all sentient species in SWAPI."""

import requests


def sentientPlanets():
    """
    Return a list of names of the home planets of all sentient species in SWAPI.

    Sentience is defined by 'sentient' in 'classification' or 'designation'.
    Handles all API pagination.

    Returns:
        list: Names of the home planets. 'unknown' if homeworld is None.
    """
    url = 'https://swapi-api.hbtn.io/api/species/'
    planets = []
    while url:
        res = requests.get(url)
        if res.status_code != 200:
            break
        data = res.json()
        for item in data.get('results', []):
            classification = (item.get('classification') or '').lower()
            designation = (item.get('designation') or '').lower()
            if ('sentient' in classification or 'sentient' in designation):
                home = item.get('homeworld')
                if home:
                    home_res = requests.get(home)
                    if home_res.status_code == 200:
                        home_name = home_res.json().get('name', 'unknown')
                    else:
                        home_name = 'unknown'
                else:
                    home_name = 'unknown'
                planets.append(home_name)
        url = data.get('next')
    return planets
