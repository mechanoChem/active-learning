import numpy as np
from configparser import ConfigParser
import os
import json
import sys


class Dictionary():

    def __init__(self, input_file,originalpath=False):
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"File not found: {input_file}")

        filepath = os.path.dirname(__file__)

        with open('{}/sample.json'.format(filepath)) as json_file:
            self.dict = json.load(json_file)
        self.construct_dict(input_file)
        if originalpath:
            self.dict['Main']['dir_path'] = originalpath
        else:  
            self.dict['Main']['dir_path'] = os.path.dirname(input_file)

        self.maintain_input()
        self.construct_input_types()



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
                # print('self.dict[inputs[i][0]][inputs[i][1]]',self.dict[inputs[i][0]][inputs[i][1]])
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
        for i in range(np.shape(inputs)[0]):
            if not os.path.isabs(self.dict[inputs[i][0]][inputs[i][1]]):
                pathlocation = os.path.join( self.dict['Main']['dir_path'], self.dict[inputs[i][0]][inputs[i][1]])
                try:
                    assert os.path.exists(pathlocation)
                except AssertionError:
                    print('Path location not found:', pathlocation)
                    sys.exit(1)  # Terminate the program with a non-zero exit status
                self.dict[inputs[i][0]][inputs[i][1]] = pathlocation


    
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
            true_false += [['IDNN','idnn_hyperparameter'],['IDNN','idnn_train_new_idnn'],['IDNN','weightrecent'],['IDNN','earlystopping'] ]
            float_array_inputs += [['IDNN','loss_weights']]
            paths += [['IDNN','transforms_directory']]
            if self.dict['IDNN']['idnn_hyperparameter'] == 'True':
                int_inputs += [['IDNN_Hyperparameter','n_sets']]
                self.set_as_str_array([['IDNN_Hyperparameter','activation'],['IDNN_Hyperparameter','optimizer']],nullallowed=True)
                self.set_as_int_array([['IDNN_Hyperparameter','layers'],['IDNN_Hyperparameter','neurons'], ['IDNN_Hyperparameter','epochs'],['IDNN_Hyperparameter','batch_size']],nullallowed=True)
                self.set_as_float_array([['IDNN_Hyperparameter','dropout'],['IDNN_Hyperparameter','learning'], ['IDNN_Hyperparameter','lr_decay'],['IDNN_Hyperparameter','factor'],['IDNN_Hyperparameter','patience'],['IDNN_Hyperparameter','min_lr']],nullallowed=True)
      
        float_inputs+= [['Main','temp']]
        true_false += [['Main','data_generation']]
        if self.dict['Main']['data_generation'] == 'True':
            if self.dict['Main']['data_generation_source'] == 'CASM':
                paths+=[['CASM','casm_project_dir']]
                # assert(os.path.exists(self.dict['CASM']['casm_project_dir']))
                assert('initial_mu' in self.dict['CASM'])
                if self.dict["CASM"]["casm_version"] == '0.3.X':
                    self.dict["CASM"]["casm_version"] = 'LCO'
                # print(self.dict["CASM"]["phi"])
                float_array_inputs += [['CASM','phi']]
                int_inputs += [['CASM','n_jobs']]
                int_array_inputs += [['CASM','relevent_indices']]
            if self.dict['Main']['data_generation_source'] == 'CASM_Surrogate':
                paths += ([['CASM_Surrogate','casm_project_dir'],['CASM_Surrogate','transforms_directory']])
                assert('initial_mu' in self.dict['CASM_Surrogate'])
                assert('activation' in self.dict['CASM_Surrogate'])
                if self.dict['CASM_Surrogate']["casm_version"] == '0.3.X':
                    self.dict['CASM_Surrogate']["casm_version"] = 'LCO'
                # int_array_inputs += [['CASM_Surrogate','hidden_layers']]
                float_array_inputs += [['CASM_Surrogate','phi']]#,['CASM_Surrogate','input_shape']]
                int_inputs += [['CASM_Surrogate','n_jobs'],['CASM_Surrogate','dim']]
                int_array_inputs += [['CASM_Surrogate','relevent_indices']]

            if self.dict['Sampling_Job_Manager']['job_manager']=='slurm':
                assert('account' in self.dict['Sampling_Job_Manager'])
                assert('walltime' in self.dict['Sampling_Job_Manager'])
                assert('mem' in self.dict['Sampling_Job_Manager'])
            

        # print('exploit params',self.dict['Exploit_Parameters'])
        true_false += [['Main','restart'],['Main','input_data'],['Main','reweight'],['Explore_Parameters','sample_external'],
                       ['Exploit_Parameters','high_error'],['Exploit_Parameters','non_convexities'],
                       ['Exploit_Parameters','find_wells'],['Exploit_Parameters','lowest_free_energy'],['Exploit_Parameters','sensitivity'],['Exploit_Parameters','qbc']]
        str_array_inputs += [['Main','input_alias'],['Main','output_alias']]
        int_inputs += [['Main','iterations'],['Main','seed'],['Main','prediction_points'],['Explore_Parameters','global_points']]
        float_inputs += [['Main','reweight_alpha']]

        if self.dict['Exploit_Parameters']['non_convexities'] == 'True':
            int_array_inputs += [['Exploit_Parameters','non_convexities_repeat'],['Exploit_Parameters','non_convexities_repeat_points']]
            float_inputs+= [['Exploit_Parameters','non_convexities_perturb_magnitude']]
        if self.dict['Exploit_Parameters']['high_error'] == 'True':
            int_array_inputs += [['Exploit_Parameters','high_error_repeat'],['Exploit_Parameters','high_error_repeat_points']]
            float_inputs+= [['Exploit_Parameters','high_error_perturb_magnitude']]
        if self.dict['Exploit_Parameters']['find_wells'] == 'True':
            int_array_inputs += [['Exploit_Parameters','wells_repeat'],['Exploit_Parameters','wells_repeat_points']]
            float_inputs+= [['Exploit_Parameters','wells_perturb_magnitude']]
        if self.dict['Exploit_Parameters']['sensitivity'] == 'True':
            int_array_inputs += [['Exploit_Parameters','sensitivity_repeat'],['Exploit_Parameters','sensitivity_repeat_points']]
            float_inputs+= [['Exploit_Parameters','sensitivity_perturb_magnitude']]
        if self.dict['Exploit_Parameters']['qbc'] == 'True':
            int_array_inputs += [['Exploit_Parameters','qbc_repeat'],['Exploit_Parameters','qbc_repeat_points']]
            float_inputs+= [['Exploit_Parameters','qbc_perturb_magnitude']]

        # if self.dict['Exploit_Parameters']['qbc'] == 'True':
        #     int_inputs += [['Exploit_Parameters','qbc_points']]

        if self.dict['Exploit_Parameters']['lowest_free_energy'] == 'True':
            int_array_inputs+= [['Exploit_Parameters','lowest_repeat'],['Exploit_Parameters','lowest_repeat_points']]
            paths += [['Exploit_Parameters','lowest_file']]
            float_inputs+= [['Exploit_Parameters','lowest_perturb_magnitude']]



        if self.dict['Explore_Parameters']['sample_external'] == 'True':
            int_inputs += [['Explore_Parameters','external_points']]
            float_inputs += [['Explore_Parameters','external_perturb_magnitude']]
            paths += ([['Explore_Parameters','external']])

        # if self.dict['Explore_Parameters']['sample_known_wells'] == 'True':
        #     int_inputs += [['Explore_Parameters','wells_points']]
        #     paths += ([['Explore_Parameters','wells']])
        # if self.dict['Explore_Parameters']['sample_known_vertices'] == 'True':
        #     int_inputs += [['Explore_Parameters','vertices_points']]
        #     paths += ([['Explore_Parameters','vertices']])



        
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
            # print('input',input)
            # print('dimensions',self.dict[input]['dimensions'])
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
            # rows_to_keep = [0, 29, 30, 31]
            # rows_to_keep = [0,1,2,3,4,5,6]
            # if self.dict['Main']['data_generation_source'] == 'CASM':
            relevent_indices = self.dict[self.dict['Main']['data_generation_source']]['relevent_indices']
            Q = np.loadtxt(self.dict['Sampling']['continuous_dependent'][domain]['filepath'])
            invQ = np.linalg.inv(Q)[:,relevent_indices]
            self.dict['Sampling']['continuous_dependent'][domain]['Q'] = Q[:,relevent_indices]
            self.dict['Sampling']['continuous_dependent'][domain]['n_planes'] = np.vstack((invQ,-invQ))
            self.dict['Sampling']['continuous_dependent'][domain]['c_planes']= np.hstack((np.ones(invQ.shape[0]),np.zeros(invQ.shape[0])))
            self.dict['Sampling']['continuous_dependent'][domain]['invQ']= invQ

            # print(np.shape(self.dict['Sampling']['continuous_dependent'][domain]['Q'] ))

            # print('Q from file', np.shape(Q))
            # print('invQ',np.shape(invQ))
            # print('final Q', np.shape(self.dict['Sampling']['continuous_dependent'][domain]['Q']))
            # print("n_planes",self.dict['Sampling']['continuous_dependent'][domain]['n_planes'] )
            # print("c_planes",self.dict['Sampling']['continuous_dependent'][domain]['c_planes'] )
        

        # input_alias - the order that data appears in input/sampled_data
        # training_order - the order that data appears to train - input, input_non_derivative, output
        # should be reordered in idnn_model

        
        return True
    
    def construct_input_types(self):

        [self.input_alias] = self.get_individual_keys('Main',['input_alias'])
        
        self.dict['Ordering']={}
        self.type_of_input = np.array([]) #type of input: 0 -continuous_dependent, 1 - continuos_independent, 2- discrete 
        self.model_order = []
        j = 0
        k = 0
        for i in range(np.size(self.input_alias)):
            [domaintype] = self.get_individual_keys(self.input_alias[i],['domain_type'])
            if domaintype == 'continuous_dependent':
                self.type_of_input = np.hstack((self.type_of_input,np.zeros(1)))
            if domaintype == 'continuous_independent':
                self.type_of_input = np.hstack((self.type_of_input,np.ones(np.size(1))))
            if domaintype == 'discrete':
                self.type_of_input = np.hstack((self.type_of_input,2*np.ones(1)))
            [modeltype] = self.get_individual_keys(self.input_alias[i],['derivative_dim'])
            if modeltype:
                self.model_order.append([1,j])
                j+=1
            else:
                self.model_order.append([0,k])
                k+=1

        self.dict['Ordering']['type_of_input']=self.type_of_input 
        self.dict['Ordering']['model_order']=self.model_order


        # for x in self.sampling_dict['continuous_dependent']:
        #     self.type_of_input = np.hstack((self.type_of_input,np.zeros(1)))
        # for y in self.sampling_dict['continuous_independent']:
        #     self.type_of_input = np.hstack((self.type_of_input,np.ones(np.size(1))))
        # for z in self.sampling_dict['discrete']:
        #     self.type_of_input = np.hstack((self.type_of_input,2*np.ones(1)))
        # #reorder