import numpy as np
from configparser import ConfigParser
import os


class Dictionary():

    def __init__(self, input_file):
        self.construct_dict(input_file)
        self.maintain_input()
        self.dict['Overview']['dir_path'] = os.path.dirname(input_file)
        # print(self.dict['Overview'])
        # try:
        #     self.maintain_input()
        # except:
        #     print('Missing input data')


    def construct_dict(self,input_file):
        self.config = ConfigParser()
        self.config.read(input_file)
        self.sections = self.config.sections()
        self.dict = {}
        for sec in self.sections:
            self.dict[sec] = {}
            self.dict[sec].update(dict(self.config.items(sec)))



    def get_individual_keys(self,keylist):
        values = []
        for key in keylist:
            for sec in self.sections:
                if key in self.dict[sec]:
                    values.append(self.dict[sec][key])
        return values

    def get_category_values(self,category):
        return self.dict[category].values()

    def get_category(self,category):
        return self.dict[category]
    
    def set_as_int(self, inputs):
        for i in range(np.shape(inputs)[0]):
            self.dict[inputs[i][0]][inputs[i][1]] = int(self.dict[inputs[i][0]][inputs[i][1]])
    
    def set_as_int_array(self,inputs):
        for i in range(np.shape(inputs)[0]):
            self.dict[inputs[i][0]][inputs[i][1]] =[int(p) for p in self.dict[inputs[i][0]][inputs[i][1]].split(',')]

    def set_as_float(self, inputs):
        for i in range(np.shape(inputs)[0]):
            self.dict[inputs[i][0]][inputs[i][1]] = float(self.dict[inputs[i][0]][inputs[i][1]])
    
    def set_as_float_array(self,inputs):
        for i in range(np.shape(inputs)[0]):
            self.dict[inputs[i][0]][inputs[i][1]] =[float(p) for p in self.dict[inputs[i][0]][inputs[i][1]].split(',')]
    
    def set_as_str_array(self,inputs):
        for i in range(np.shape(inputs)[0]):
            self.dict[inputs[i][0]][inputs[i][1]] =[str(p) for p in self.dict[inputs[i][0]][inputs[i][1]].split(',')]
    

    def set_as_true_false(self,inputs):
        for i in range(np.shape(inputs)[0]):
            if  self.dict[inputs[i][0]][inputs[i][1]] == 'True':
                 self.dict[inputs[i][0]][inputs[i][1]] = True
            else:
                 self.dict[inputs[i][0]][inputs[i][1]] = False


    
    def maintain_input(self):
        int_inputs = []
        int_array_inputs = []
        float_inputs = []
        float_array_inputs = []
        str_array_inputs = []
        true_false = []
        if self.dict['Overview']['model'] == 'IDNN':
            int_inputs +=[ ['Neural Network','epochs'],['Neural Network','batch_size'],['Neural Network','n_sets'] ]
            int_array_inputs += [['Neural Network','hidden_units'],['Neural Network','layers'],['Neural Network','neurons']]
            str_array_inputs += [['Neural Network','activation']]
            float_array_inputs += [['Neural Network','learningrate']]
            float_inputs += [['Neural Network','dropout'],['Neural Network','learning'],['Neural Network','validation_split']]
            assert('optimizer' in self.dict['Neural Network'])

        true_false += [['Overview','data_generation']]
        if self.dict['Overview']['data_generation'] == 'True':
            true_false += [['CASM Data Generation','surrogate']]
            if self.dict['Overview']['data_generation_source'] == 'CASM':
                assert(os.path.exists(self.dict['CASM Data Generation']['casm_project_dir']))
                if self.dict['CASM Data Generation']['job_manager']=='slurm':
                    assert('account' in self.dict['CASM Data Generation'])
                    assert('walltime' in self.dict['CASM Data Generation'])
                    assert('mem' in self.dict['CASM Data Generation'])
                assert('initial_mu' in self.dict['CASM Data Generation'])
                float_array_inputs += [['CASM Data Generation','phi']]
                int_inputs += [['CASM Data Generation','n_jobs']]


        true_false += [['Overview','restart'],['Overview','input_data'],['Exploit Parameters','hessian'],['Exploit Parameters','high_error'],['Sampling Domain','sample_vertices']]
        str_array_inputs += [['Overview','input_alias'],['Overview','output_alias'],['Sampling Domain','sample_wells']]
        int_inputs += [['Overview','iterations'],['Overview','seed'],['Sampling Domain','global_points'],['Hyperparameter','n_sets']]


        if self.dict['Exploit Parameters']['hessian'] == 'True':
            int_array_inputs += [['Exploit Parameters','hessian_repeat'],['Exploit Parameters','hessian_repeat_points']]
        if self.dict['Exploit Parameters']['high_error'] == 'True':
            int_array_inputs += [['Exploit Parameters','high_error_repeat'],['Exploit Parameters','high_error_repeat_points']]

        str_array_inputs += [['Training','loss']]
        float_array_inputs +=[['Training','loss_weights']]
        float_inputs += [['Training','lr_decay'],['Training','factor'],['Training','patience']]
        int_array_inputs += [['Sampling Domain','domain'],['Hyperparameter','layers_range'],['Hyperparameter','neurons_range']]
        # float_array_inputs += [['Sampling Domain','x0']]

        

        self.set_as_int(int_inputs)
        self.set_as_int_array(int_array_inputs)
        self.set_as_float(float_inputs)
        self.set_as_float_array(float_array_inputs)
        self.set_as_str_array(str_array_inputs)
        self.set_as_true_false(true_false)



        input_dim = 0
        derivative_dim = 0
        output_dim = 0 #np.size(self.dict['Overview']['output_alias'])
        self.dict['Sampling'] = {}
        self.dict['Sampling']['continuous_dependent'] = {}
        self.dict['Sampling']['continuous_independent'] = {}
        self.dict['Sampling']['discrete'] = {}
        for input in self.dict['Overview']['input_alias']:
            self.set_as_int([[input,'dimensions']])
            if self.dict[input]['domain_type']=='continuous_dependent':
                domain = self.dict[input]['domain']
                if domain not in self.dict['Sampling']['continuous_dependent']:
                    self.dict['Sampling']['continuous_dependent'][domain] = {'values': [input], 'dim':self.dict[input]['dimensions']}
                    test_set = self.dict[domain]['test_set']
                    self.dict['Sampling']['continuous_dependent'][domain]['type'] = test_set
                    if test_set == 'billiardwalk':
                        self.set_as_float_array([[domain,'x0']])
                        self.dict['Sampling']['continuous_dependent'][domain]['x0'] = self.dict[domain]['x0']
                    elif test_set == 'sobol':
                        self.dict['Sampling']['continuous_dependent'][domain]['x_bounds'] = self.dict[domain]['x_bounds']
                    self.dict['Sampling']['continuous_dependent'][domain]['filepath'] = self.dict[domain]['filepath']
                else: 
                    self.dict['Sampling']['continuous_dependent'][domain]['values'] = [input]
                    self.dict['Sampling']['continuous_dependent'][domain]['dim'] += self.dict[input]['dimensions']  
            else:
                self.set_as_float_array([[input,'domain']])
                self.dict['Sampling'][self.dict[input]['domain_type']][input] = self.dict[input]['domain']
            input_dim +=  self.dict[input]['dimensions']
            self.set_as_float_array([[input,'adjust']])
            self.set_as_true_false([[input,'derivative_dim']] )
            if self.dict[input]['derivative_dim']:
                derivative_dim += self.dict[input]['dimensions']
        
        for output in self.dict['Overview']['output_alias']:
            self.set_as_int([[output,'derivative'],[output,'dimensions']])
            self.set_as_float_array([[output,'adjust']])
            output_dim += self.dict[output]['dimensions']




        self.dict['Overview']['Input_dim'] = input_dim
        self.dict['Overview']['Derivative_dim'] = derivative_dim    
        self.dict['Overview']['Output_dim'] = output_dim


                

        for domain in self.dict['Sampling']['continuous_dependent']:
            Q = np.loadtxt(self.dict['Sampling']['continuous_dependent'][domain]['filepath'])
            invQ = np.linalg.inv(Q)[:,:self.dict['Sampling']['continuous_dependent'][domain]['dim']]
            self.dict['Sampling']['continuous_dependent'][domain]['Q'] = Q[:self.dict['Sampling']['continuous_dependent'][domain]['dim']]
            self.dict['Sampling']['continuous_dependent'][domain]['n_planes'] = np.vstack((invQ,-invQ))
            self.dict['Sampling']['continuous_dependent'][domain]['c_planes']= np.hstack((np.ones(invQ.shape[0]),np.zeros(invQ.shape[0])))
            self.dict['Sampling']['continuous_dependent'][domain]['invQ']= invQ

        
        return True