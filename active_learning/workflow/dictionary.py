import numpy as np
from configparser import ConfigParser


class Dictionary():

    def __init__(self, input_file):
        self.construct_dict(input_file)
        self.maintain_input()


    def construct_dict(self,input_file):
        self.config = ConfigParser()
        self.config.read(input_file)
        # print(config.sections())
        self.sections = self.config.sections()
        self.dict = {}
        for sec in self.sections:
            self.dict.update(dict(self.config.items(sec)))
        # print(self.dict)
        # for key in self.dict:
        #     print(key)
        #     print(self.dict[key])
        # print('config',self.config)



    def get_individual_keys(self,keylist):
        values = []
        # print(self.dict)
        for key in keylist:
            values.append(self.dict[key])
        return values

    def get_category_values(self,category):
        values = []
        category = dict(self.config.items(category))
        for key in category:
            if key == 'x0':
                self.dict[key] = [float(p) for p in self.dict[key] .split(',')]
            if key == 'temperatures':
                self.dict[key] = [float(p) for p in self.dict[key] .split(',')]
            values.append(self.dict[key])
        return values 

    def verify_necessary_information(self,):
        return True
    
    def maintain_input(self):
        self.dict['hidden_units'] = [int(p) for p in self.dict['hidden_units'] .split(',')]
        self.dict['global_points'] = int( self.dict['global_points'] )
        self.dict['derivative_dim'] = int( self.dict['derivative_dim'] )
        # print(self.dict['hidden_units'])
        
        return True