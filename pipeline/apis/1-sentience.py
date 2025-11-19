#!/usr/bin/env python3
"""Module to find home planets of all sentient species in SWAPI."""
import requests


def sentientPlanets():
    """
    Return the list of home planet names of all sentient species.

    A sentient species has 'sentient' in its classification or designation.
    Pagination on the species endpoint is fully handled.
    """
    url = 'https://swapi-api.hbtn.io/api/species/'
    planets = []
    seen = set()

    while url:
        res = requests.get(url)
        if res.status_code != 200:
            break

        data = res.json()
        for item in data.get('results', []):
            classification = (item.get('classification') or '').lower()
            designation = (item.get('designation') or '').lower()

            if 'sentient' not in classification and 'sentient' not in designation:
                continue

            home_url = item.get('homeworld')
            if not home_url:
                continue

            if home_url in seen:
                continue
            seen.add(home_url)

            home_res = requests.get(home_url)
            if home_res.status_code != 200:
                continue

            name = home_res.json().get('name')
            if not name:
                continue

            planets.append(name)

        url = data.get('next')

    return planets
