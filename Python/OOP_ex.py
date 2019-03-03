import pandas as pd
import numpy as np


class Town():
    # defining attributes of class
    def __init__(self, num, name):
        self.rooms = num
        self.host = name
        self.description = None
        self.neighbors = {}

    # defining methods of class
    def set_description(self, text):
        self.description = text

    def get_description(self):
        return self.description

    def bridge(self, neighbor, direction):
        self.neighbors[direction] = neighbor
        print("{} lives in the {}".format(self.host, direction))
