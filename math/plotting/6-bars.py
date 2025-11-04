#!/usr/bin/env python3
"""
Module that plots a stacked bar graph showing the number
of different fruits per person.
"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Create a stacked bar chart representing the quantity of
    various fruits each person has.

    - `fruit` is a 4x3 matrix: rows = apples, bananas, oranges,
      peaches; columns = Farrah, Fred, Felicia.
    - Bars are stacked in the same order as rows (bottom -> top).
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    people = ['Farrah', 'Fred', 'Felicia']
    ind = np.arange(len(people))
    width = 0.5

    # Split rows for clarity
    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    # Create figure
    plt.figure(figsize=(6.4, 4.8))

    # Plot stacked bars (bottom offsets provided by cumulative sums)
    plt.bar(ind, apples, width, color='red', label='apples')
    plt.bar(ind, bananas, width, bottom=apples, color='yellow',
            label='bananas')
    plt.bar(ind, oranges, width, bottom=apples + bananas,
            color='#ff8000', label='oranges')
    plt.bar(ind, peaches, width, bottom=apples + bananas + oranges,
            color='#ffe5b4', label='peaches')

    # Labels, ticks, legend and limits
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.xticks(ind, people)
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()
    plt.show()
