import numpy as np
from configparser import ConfigParser
import os


class Dictionary():

    def __init__(self, input_file):
        self.construct_dict(input_file)
        self.maintain_input()
        # try:
        #     self.maintain_input()
        # except:
        #     print('Missing input data')


    def construct_dict(self,input_file):
        self.config = ConfigParser()
        self.config.read(input_file)
        # print(config.sections())]
        # self.maintain_input()
        self.sections = self.config.sections()
        # print(self.sections)
        self.dict = {}
        for sec in self.sections:
            self.dict[sec] = {}
            self.dict[sec].update(dict(self.config.items(sec)))
            # self.dict.update(dict(temp_dict))
        # print(self.dict
        # for key in self.dict:
        #     print(key)
        #     print(self.dict[key])
        # print('config',self.config)



    def get_individual_keys(self,keylist):
        values = []
        # print(self.dict)
        for key in keylist:
            for sec in self.sections:
                if key in self.dict[sec]:
                    values.append(self.dict[sec][key])
        return values

    def get_category_values(self,category):
        # values = []
        # category = dict(self.config.items(category))
        # for key in category:
        #     values.append(self.dict[key])
        return self.dict[category].values()


    
    def maintain_input(self):
        if self.dict['Overview']['model'] == 'IDNN':
            self.dict['Neural Network']['epochs'] = int(self.dict['Neural Network']['epochs'])
            self.dict['Neural Network']['batch_size'] = int(self.dict['Neural Network']['batch_size'])
            self.dict['Neural Network']['activation'] = [str(p) for p in self.dict['Neural Network']['activation'].split(',')]
            self.dict['Neural Network']['hidden_units'] = [int(p) for p in self.dict['Neural Network']['hidden_units'].split(',')]
            self.dict['Neural Network']['n_sets'] = int(self.dict['Neural Network']['n_sets'])
            self.dict['Neural Network']['learningrate'] = [float(p) for p in self.dict['Neural Network']['learningrate'].split(',')]
            self.dict['Neural Network']['layers'] = [int(p) for p in self.dict['Neural Network']['layers'].split(',')]
            self.dict['Neural Network']['neurons'] = [int(p) for p in self.dict['Neural Network']['neurons'].split(',')]
            self.dict['Neural Network']['dropout'] = float(self.dict['Neural Network']['dropout'])
            self.dict['Neural Network']['learning'] = float(self.dict['Neural Network']['learning'])
            assert('optimizer' in self.dict['Neural Network'])

        if self.dict['Overview']['data_generation'] == 'True':
            self.dict['Overview']['data_generation'] = True
            if self.dict['CASM Data Generation']['surrogate']== 'True':
                self.dict['CASM Data Generation']['surrogate']=True
                # self.Hidden_Layers = self.dict['DATA_GENERATION']['Hidden_Layers']
                # self.data_gen_activation = self.dict['DATA_GENERATION']['Activation']
                # self.Input_Shape =  self.dict['DATA_GENERATION']['Input_Shape']
            else:
                self.dict['CASM Data Generation']['surrogate']=False

            if self.dict['Overview']['data_generation_source'] == 'CASM':
                assert(os.path.exists(self.dict['CASM Data Generation']['casm_project_dir']))
                if self.dict['CASM Data Generation']['job_manager']=='slurm':
                    assert('account' in self.dict['CASM Data Generation'])
                    assert('walltime' in self.dict['CASM Data Generation'])
                    assert('mem' in self.dict['CASM Data Generation'])
                assert('initial_mu' in self.dict['CASM Data Generation'])
                self.dict['CASM Data Generation']['phi'] = np.array([float(p) for p in self.dict['CASM Data Generation']['phi'].split(',')])
                self.dict['CASM Data Generation']['n_jobs'] = int(self.dict['CASM Data Generation']['n_jobs'])
        else:
            self.dict['Overview']['data_generation'] = False


        if self.dict['Overview']['restart'] == 'True':
            self.dict['Overview']['restart'] = True
        else:
            self.dict['Overview']['restart'] = False


        if self.dict['Overview']['input_data'] == 'True':
            self.dict['Overview']['input_data'] = True
        else:
            self.dict['Overview']['input_data'] = False

        self.dict['Overview']['input_dim'] = int(self.dict['Overview']['input_dim'])
        self.dict['Overview']['output_dim'] = int(self.dict['Overview']['output_dim'])
        self.dict['Overview']['derivative_dim'] = int(self.dict['Overview']['derivative_dim'])
        self.dict['Overview']['iterations'] = int(self.dict['Overview']['iterations'])
        self.dict['Overview']['seed'] = int(self.dict['Overview']['seed'])
        self.dict['Overview']['temperatures'] = [float(p) for p in self.dict['Overview']['temperatures'].split(',')]


        if self.dict['Exploit Parameters']['hessian'] == 'True':
            self.dict['Exploit Parameters']['hessian'] = True
            self.dict['Exploit Parameters']['hessian_repeat'] = [int(p) for p in self.dict['Exploit Parameters']['hessian_repeat'].split(',')]
            self.dict['Exploit Parameters']['hessian_repeat_points'] = [int(p) for p in self.dict['Exploit Parameters']['hessian_repeat_points'].split(',')]
        else:
            self.dict['Exploit Parameters']['hessian'] = False
        if self.dict['Exploit Parameters']['high_error'] == 'True':
            self.dict['Exploit Parameters']['high_error'] = True
            self.dict['Exploit Parameters']['high_error_repeat'] = [int(p) for p in self.dict['Exploit Parameters']['high_error_repeat'].split(',')]
            self.dict['Exploit Parameters']['high_error_repeat_points'] = [int(p) for p in self.dict['Exploit Parameters']['high_error_repeat_points'].split(',')]
        else:
            self.dict['Exploit Parameters']['high_error'] = False

        if self.dict['Sampling Domain']['sample_vertices']=='True':
            self.dict['Sampling Domain']['sample_vertices']=True
        else:
            self.dict['Sampling Domain']['sample_vertices']=False
        
        self.dict['Training']['lr_decay']=  float(self.dict['Training']['lr_decay'])

        self.dict['Sampling Domain']['sample_wells']=[str(p) for p in self.dict['Sampling Domain']['sample_wells'].split(',')]

        self.dict['Sampling Domain']['domain'] = [int(p) for p in self.dict['Sampling Domain']['domain'].split(',')]
        self.dict['Sampling Domain']['global_points'] = int(self.dict['Sampling Domain']['global_points'])
        self.dict['Sampling Domain']['x0'] = [float(p) for p in self.dict['Sampling Domain']['x0'].split(',')]
        


        
        return True