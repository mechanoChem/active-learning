import numpy as np
from configparser import ConfigParser


class Dictionary():

    def __init__(self, input_file):
        self.config = self.construct_dict(input_file)


    def construct_dict(input_file):
        config = ConfigParser()
        config.read(input_file)
        return config
        #print(config.sections())
        # d = {}
        # with open("dict.txt") as f:
        #     for line in f:
        #         (key, val) = line.split()
        #         d[int(key)] = val

        # print (d)
    
    def get_individual_keys(keylist):
        values = []
        return values

    def get_category_values(category):
        values = []
        return values 

    def verify_necessary_information():
        return True