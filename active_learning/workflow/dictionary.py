import numpy as np
from configparser import ConfigParser
import os
import json
import sys


class Dictionary():

    def __init__(self, input_file):

        with open('../../active_learning/workflow/sample.json') as json_file:
            self.dict = json.load(json_file)
        # print(self.dict)
        self.construct_dict(input_file)
        self.dict['Main']['dir_path'] = os.path.dirname(input_file)
        self.maintain_input()

        # try:
        #     self.maintain_input()
        # except:
        #     print('Missing input data')
        #     return -1
        
        # return 0


    def construct_dict(self,input_file):
        self.config = ConfigParser()
        self.config.read(input_file)
        self.sections = self.config.sections()
        for sec in self.sections:
            if sec in self.dict:
                for item in self.config.items(sec):
                    self.dict[sec][item[0]] = item[1]
            else:
                self.dict[sec] = {}
                self.dict[sec].update(dict(self.config.items(sec)))


    def get_individual_keys(self,category,keylist):
        return [self.dict[category][key] for key in keylist]

    def get_category_values(self,category):
        return self.dict[category].values()

    def get_category(self,category):
        return self.dict[category]
    
    def set_as_int(self, inputs,nullallowed=False):
        for i in range(np.shape(inputs)[0]):
            if self.dict[inputs[i][0]][inputs[i][1]] != None:
                self.dict[inputs[i][0]][inputs[i][1]] = int(self.dict[inputs[i][0]][inputs[i][1]])
            elif not nullallowed:
                raise Exception('Input',inputs[i],'is null')
    
    def set_as_int_array(self,inputs,nullallowed=False):
        for i in range(np.shape(inputs)[0]):
            if self.dict[inputs[i][0]][inputs[i][1]] != None:
                self.dict[inputs[i][0]][inputs[i][1]] =[int(p) for p in self.dict[inputs[i][0]][inputs[i][1]].split(',')]
            elif not nullallowed:
                raise Exception('Input',inputs[i],'is null')

    def set_as_float(self, inputs,nullallowed=False):
        for i in range(np.shape(inputs)[0]):
            if self.dict[inputs[i][0]][inputs[i][1]] != None:
                self.dict[inputs[i][0]][inputs[i][1]] = float(self.dict[inputs[i][0]][inputs[i][1]])
            elif not nullallowed:
                raise Exception('Input',inputs[i],'is null')
    
    def set_as_float_array(self,inputs,nullallowed=False):
        for i in range(np.shape(inputs)[0]):
            if self.dict[inputs[i][0]][inputs[i][1]] != None:
                self.dict[inputs[i][0]][inputs[i][1]] =[float(p) for p in self.dict[inputs[i][0]][inputs[i][1]].split(',')]
            elif not nullallowed:
                raise Exception('Input',inputs[i],'is null')
    
    def set_as_str_array(self,inputs,nullallowed=False):
        for i in range(np.shape(inputs)[0]):
            if self.dict[inputs[i][0]][inputs[i][1]] != None:
                self.dict[inputs[i][0]][inputs[i][1]] =[str(p) for p in self.dict[inputs[i][0]][inputs[i][1]].split(',')]
            elif not nullallowed:
                raise Exception('Input',inputs[i],'is null')
    

    def set_as_true_false(self,inputs):
        for i in range(np.shape(inputs)[0]):
            if  self.dict[inputs[i][0]][inputs[i][1]] == 'True':
                 self.dict[inputs[i][0]][inputs[i][1]] = True
            else:
                 self.dict[inputs[i][0]][inputs[i][1]] = False


    def verifypath(self,inputs):
        # print('directory path',self.dict['Main']['dir_path'] )
        # print(inputs)
        for i in range(np.shape(inputs)[0]):
            # print('verifypath')
            # print('original path',self.dict[inputs[i][0]][inputs[i][1]])
            # print("ispath", os.path.isabs(self.dict[inputs[i][0]][inputs[i][1]]))
            if not os.path.isabs(self.dict[inputs[i][0]][inputs[i][1]]):
                pathlocation = os.path.join( self.dict['Main']['dir_path'], self.dict[inputs[i][0]][inputs[i][1]])
                assert(os.path.exists(pathlocation))
                # print(pathlocation)
                self.dict[inputs[i][0]][inputs[i][1]] = pathlocation
            # print("is new path", os.path.isabs(self.dict[inputs[i][0]][inputs[i][1]]))
            # assert(0==1)





    
    def maintain_input(self):
        int_inputs = []
        int_array_inputs = []
        float_inputs = []
        float_array_inputs = []
        str_array_inputs = []
        true_false = []
        paths = []
        if self.dict['Main']['model'] == 'IDNN':
            int_inputs +=[  ['IDNN','layers'], ['IDNN','neurons'], ['IDNN','epochs'],['IDNN','batch_size'] ]
            str_array_inputs += [['IDNN','activation'],['IDNN','loss']]
            float_inputs += [['IDNN','dropout'],['IDNN','learning'],['IDNN','lr_decay'],['IDNN','factor'],['IDNN','patience'],['IDNN','min_lr'], ['IDNN','validation_split']]
            assert('optimizer' in self.dict['IDNN'])
            true_false += [['IDNN','idnn_hyperparameter']]
            float_array_inputs += [['IDNN','loss_weights']]
            paths += [['IDNN','transforms_directory']]
            if self.dict['IDNN']['idnn_hyperparameter'] == 'True':
                int_inputs += [['IDNN_Hyperparameter','n_sets']]
                str_array_inputs += [['IDNN_Hyperparameter','activation'],['IDNN_Hyperparameter','optimizer']]
                self.set_as_int_array([['IDNN_Hyperparameter','layers'],['IDNN_Hyperparameter','neurons'], ['IDNN_Hyperparameter','epochs'],['IDNN_Hyperparameter','batch_size']],nullallowed=True)
                self.set_as_float_array([['IDNN_Hyperparameter','dropout'],['IDNN_Hyperparameter','learning'], ['IDNN_Hyperparameter','lr_decay'],['IDNN_Hyperparameter','factor'],['IDNN_Hyperparameter','patience'],['IDNN_Hyperparameter','min_lr']],nullallowed=True)
      

        true_false += [['Main','data_generation']]
        if self.dict['Main']['data_generation'] == 'True':
            if self.dict['Main']['data_generation_source'] == 'CASM':
                paths+=[['CASM','casm_project_dir']]
                # assert(os.path.exists(self.dict['CASM']['casm_project_dir']))
                assert('initial_mu' in self.dict['CASM'])
                if self.dict["CASM"]["casm_version"] == '0.3.X':
                    self.dict["CASM"]["casm_version"] = 'LCO'
                float_array_inputs += [['CASM','phi']]
                int_inputs += [['CASM','n_jobs']]
            if self.dict['Main']['data_generation_source'] == 'CASM_Surrogate':
                paths += ([['CASM_Surrogate','casm_project_dir'],['CASM_Surrogate','transforms_directory']])
                assert('initial_mu' in self.dict['CASM_Surrogate'])
                assert('activation' in self.dict['CASM_Surrogate'])
                if self.dict['CASM_Surrogate']["casm_version"] == '0.3.X':
                    self.dict['CASM_Surrogate']["casm_version"] = 'LCO'
                # int_array_inputs += [['CASM_Surrogate','hidden_layers']]
                float_array_inputs += [['CASM_Surrogate','phi']]#,['CASM_Surrogate','input_shape']]
                int_inputs += [['CASM_Surrogate','n_jobs'],['CASM_Surrogate','dim']]
                if self.dict['CASM_Surrogate']["version"] == '0.3.X':
                    self.dict['CASM_Surrogate']["version"] = 'LCO'

            if self.dict['Sampling_Job_Manager']['job_manager']=='slurm':
                assert('account' in self.dict['Sampling_Job_Manager'])
                assert('walltime' in self.dict['Sampling_Job_Manager'])
                assert('mem' in self.dict['Sampling_Job_Manager'])
            


        true_false += [['Main','restart'],['Main','input_data'],['Exploit_Parameters','hessian'],
                       ['Exploit_Parameters','high_error'],['Explore_Parameters','sample_known_wells'],
                       ['Explore_Parameters','sample_known_vertices'],['Exploit_Parameters','hessian'],
                       ['Exploit_Parameters','high_error'],['Exploit_Parameters','find_wells']]
        str_array_inputs += [['Main','input_alias'],['Main','output_alias']]
        int_inputs += [['Main','iterations'],['Main','seed'],['Explore_Parameters','global_points']]


        if self.dict['Exploit_Parameters']['hessian'] == 'True':
            int_array_inputs += [['Exploit_Parameters','hessian_repeat'],['Exploit_Parameters','hessian_repeat_points']]
        if self.dict['Exploit_Parameters']['high_error'] == 'True':
            int_array_inputs += [['Exploit_Parameters','high_error_repeat'],['Exploit_Parameters','high_error_repeat_points']]
        if self.dict['Exploit_Parameters']['find_wells'] == 'True':
            int_array_inputs += [['Exploit_Parameters','wells_repeat'],['Exploit_Parameters','wells_repeat_points']]

        if self.dict['Explore_Parameters']['sample_known_wells'] == 'True':
            int_inputs += [['Explore_Parameters','wells_points']]
            paths += ([['Explore_Parameters','wells']])
        if self.dict['Explore_Parameters']['sample_known_vertices'] == 'True':
            int_inputs += [['Explore_Parameters','vertices_points']]
            paths += ([['Explore_Parameters','vertices']])


        # str_array_inputs += [['Training','loss']]
        # float_array_inputs +=[['Training','loss_weights']]
        # float_inputs += [['Training','lr_decay'],['Training','factor'],['Training','patience']]
        
        
        
        # int_array_inputs += [['Hyperparameter','layers_range'],['Hyperparameter','neurons_range']]

        
        if self.dict['Main']['restart'] == 'True':
            int_inputs += [['Restart','rnd']]

        self.set_as_int(int_inputs)
        self.set_as_int_array(int_array_inputs)
        self.set_as_float(float_inputs)
        self.set_as_float_array(float_array_inputs)
        self.set_as_str_array(str_array_inputs)
        self.set_as_true_false(true_false)
        self.verifypath(paths)



        input_dim = 0
        derivative_dim = 0
        output_dim = 0 #np.size(self.dict['Main']['output_alias'])
        self.dict['Sampling'] = {}
        self.dict['Sampling']['continuous_dependent'] = {}
        self.dict['Sampling']['continuous_independent'] = {}
        self.dict['Sampling']['discrete'] = {}
        for input in self.dict['Main']['input_alias']:
            self.set_as_int([[input,'dimensions']])
            if self.dict[input]['domain_type']=='continuous_dependent':
                domain = self.dict[input]['domain']
                if domain not in self.dict['Sampling']['continuous_dependent']:
                    self.dict['Sampling']['continuous_dependent'][domain] = {'values': [input], 'dim':self.dict[input]['dimensions']}
                    space_filling_method = self.dict[domain]['space_filling_method']
                    self.dict['Sampling']['continuous_dependent'][domain]['type'] = space_filling_method
                    if space_filling_method == 'billiardwalk':
                        self.set_as_float_array([[domain,'x0']])
                        self.dict['Sampling']['continuous_dependent'][domain]['x0'] = self.dict[domain]['x0']
                    elif space_filling_method == 'sobol':
                        self.dict['Sampling']['continuous_dependent'][domain]['x_bounds'] = self.dict[domain]['x_bounds']
                    self.verifypath([[domain,'filepath']])
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
        
        for output in self.dict['Main']['output_alias']:
            self.set_as_int([[output,'derivative'],[output,'dimensions']])
            self.set_as_float_array([[output,'adjust']])
            output_dim += self.dict[output]['dimensions']




        self.dict['Main']['input_dim'] = input_dim
        self.dict['Main']['derivative_dim'] = derivative_dim    
        self.dict['Main']['output_dim'] = output_dim


                

        for domain in self.dict['Sampling']['continuous_dependent']:
            Q = np.loadtxt(self.dict['Sampling']['continuous_dependent'][domain]['filepath'])
            invQ = np.linalg.inv(Q)[:,:self.dict['Sampling']['continuous_dependent'][domain]['dim']]
            self.dict['Sampling']['continuous_dependent'][domain]['Q'] = Q[:self.dict['Sampling']['continuous_dependent'][domain]['dim']]
            self.dict['Sampling']['continuous_dependent'][domain]['n_planes'] = np.vstack((invQ,-invQ))
            self.dict['Sampling']['continuous_dependent'][domain]['c_planes']= np.hstack((np.ones(invQ.shape[0]),np.zeros(invQ.shape[0])))
            self.dict['Sampling']['continuous_dependent'][domain]['invQ']= invQ

        
        return True