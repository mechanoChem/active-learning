import os, sys
import numpy as np
from datetime import datetime
import shutil, json
from shutil import copyfile
from tensorflow import keras
from active_learning.workflow.dictionary import Dictionary
from active_learning.workflow.hp_search import hyperparameterSearch
from active_learning.data_recommended.DataRecommender import DataRecommender
from active_learning.data_collector.CASM_Sampling import CASM_Sampling


class Workflow():

    def __init__(self,input_path):
        self.input_path = input_path
        self.dict = Dictionary(input_path)
        # print(self.dict.get_category_values('Overview'))
        [self.Model_type,    
         self.Data_Generation, self.Data_Generation_Source, 
         self.restart, self.Input_data, self.Input_alias, 
         self.Output_alias,self.Iterations, self.OutputFolder, 
         self.seed,self.input_dim,self.output_dim,self.derivative_dim,self.config_path] = self.dict.get_category_values('Main')
        
        
        [self.N_global_pts, self.sample_known_wells,self.wells,
         self.wells_points,self.sample_known_vertices,self.vertices, 
         self.vertice_points] = self.dict.get_category_values('Explore_Parameters')

        [self.sample_hessian,self.hessian_repeat, self.hessian_repeat_points,self.sample_high_error,
        self.high_error_repeat, self.high_error_repeat_points, self.find_wells, self.wells_repeat,self.wells_repeat_points] = self.dict.get_category_values('Exploit_Parameters')
        if self.restart==False:
            self.rnd=0
            if  os.path.exists(self.OutputFolder + 'training'):
                shutil.rmtree(self.OutputFolder +'training')
            if  os.path.exists(self.OutputFolder +'data/data_recommended'):
                shutil.rmtree(self.OutputFolder +'data/data_recommended')
            if  os.path.exists(self.OutputFolder +'data/data_sampled'):
                shutil.rmtree(self.OutputFolder +'data/data_sampled')
            if  os.path.exists(self.OutputFolder +'data/outputFiles'):
                shutil.rmtree(self.OutputFolder +'data/outputFiles')
            if not os.path.exists(self.OutputFolder):
                os.mkdir(self.OutputFolder)
            if not os.path.exists(self.OutputFolder+'data'):
                os.mkdir(self.OutputFolder+'data')
            os.mkdir(self.OutputFolder +'training')
            os.mkdir(self.OutputFolder +'data/data_recommended')
            os.mkdir(self.OutputFolder +'data/data_sampled')
            os.mkdir(self.OutputFolder +'data/outputFiles')
        if self.Input_data:
            self.step = 'Model_training'
            self.handle_input_data(self.Input_data)
        else:
            self.step = 'Explorative'
        self.construct_model()
        if self.restart==True:
            self.read_restart()
        if self.Data_Generation:
            if self.Data_Generation_Source=='CASM' or self.Data_Generation_Source=='CASM_Surrogate':
                self.sampling = CASM_Sampling(self.model, self.dict)
        self.recommender = DataRecommender(self.model,self.dict)


    def construct_model(self):
        if self.Model_type == 'IDNN':
            [self.transform_path] = self.dict.get_individual_keys('IDNN',['transforms_directory'])
            from active_learning.model.idnn_model import IDNN_Model 
            self.model = IDNN_Model(self.dict)



    def read_restart(self):
        #  self.dict_restart = Dictionary(input_path)
         [self.rnd, self.step] = self.dict.get_category_values('Restart')
         self.model.load_model(self.rnd-1)

    # def save_restart(self):
    #     params = [self.rnd, self.step]
    #     jsonparams = json.dumps(params)
    #     with open(self.outputFolder + 'restart.json', "w") as outfile:
    #         outfile.write(jsonparams)


    def handle_input_data(self,input_file):
        input_results = np.loadtxt('input_file')
        if os.exists(self.OutputFolder +'data_sampled/allResults.txt'):
            allResults = np.loadtxt(self.OutputFolder +'data_sampled/allResults.txt')
            allResults = np.vstack(allResults,input_file)
            np.savetxt(self.OutputFolder +'data_sampled/allResults.txt',
                allResults,
                fmt='%.12f')
        else: 
            copyfile(input_file,self.OutputFolder +'data_sampled/allResults.txt')
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        copyfile(input_file, self.OutputFolder +'data_sampled/input_data_{}.txt'.format(date_time))




    def explore(self):
        if self.sample_known_wells:
            print('Sampling Wells')
            self.recommender.sample_wells(self.rnd)
        if self.sample_known_vertices:
            print('Sampling Vertices')
            self.recommender.sample_vertices(self.rnd)
        self.recommender.explore(self.rnd)


    
    def exploit(self,model):
        self.recommender.get_latest_pred(self.rnd)
        if self.sample_high_error == True:
            self.recommender.high_error(self.rnd)
        if self.sample_hessian == True:
            self.recommender.hessian(self.rnd)
        if self.find_wells:
            self.recommender.find_wells(self.rnd)



    def sample_data(self,rnd):
        command = self.sampling.construct_job(rnd)
        self.sampling.submit_job(command)
        # print('check1')
        self.sampling.write(rnd)
        # print('sampling write complete')


    def hyperparameter_search(self,rnd):

        [N_hp_sets] = self.dict.get_individual_keys('IDNN_Hyperparameter',['n_sets'])
        job_manager,account,walltime,mem, = self.dict.get_category_values('Hyperparameter_Job_Manager')
        # submit
        if self.Model_type=='IDNN':
            commands = [f"sys.path.append('{os.path.dirname(__file__)}')",
                    'from active_learning.model.idnn_model import IDNN_Model',
                    'from active_learning.workflow.dictionary import Dictionary',
                     f'dict = Dictionary("{os.path.abspath(self.input_path)}")',
                     'model  = IDNN_Model(dict)']
        training_func = 'model.train_rand_idnn'.format(self.config_path)
    
        params = hyperparameterSearch(rnd,N_hp_sets,commands,training_func, job_manager,account,walltime,mem,self.OutputFolder)
        # self.model.new_model(params)
        self.model.load_model(rnd)
    
    def IDNN_transforms(self):


        sys.path.append(self.config_path+self.transform_path)
        # sys.path.append(self.config_path)
        from TransformsModule import transforms 

        return transforms

    def better_than_prev(self,rnd):

        # params = self.model.load_data(rnd)
     
        # Get loss of current idnn
        current_loss = self.model.loss(rnd)

        # Get loss of previous idnn
        keras.backend.clear_session()
        lastmodel = self.model.load_model(rnd-1)#, custom_objects={'Transform': self.IDNN_transforms()})
        prev_loss = self.model.loss(rnd)


        # Reload current IDNN
        self.model.load_model(rnd)#, custom_objects={'Transform': self.IDNN_transforms()})
        if current_loss < prev_loss:
            return True
        else:
            return False
    
    def train(self):
        self.model.train(self.rnd)
        self.model.save_model(self.rnd)
    


        
        


    def main_workflow(self):
        # self.model = self.hyperparameter_search(self.rnd)

        #If there is no pre-existing data : we first have to do explorative predictions
        # self.hyperparameter_search(2) 
        # self.better_than_prev(1)
        if self.step == 'Explorative':
            print('Explorative Data Recommendations, round ',self.rnd,'...')
            self.explore()
            if self.Data_Generation==True:
                self.step = 'Sampling'
            else:
                self.step = 'Complete'

        if self.step == 'Sampling':
            print('Data sampling, round ',self.rnd,'...')
            self.sample_data(self.rnd)
            self.step = 'Model_training'

        

        if self.step == 'Model_training':
            print('Train surrogate model, round ',self.rnd,'...')
            self.train()
            if self.Data_Generation==False:
                self.model = self.hyperparameter_search(self.rnd)
                self.step = 'Complete'
                print('Exploitative Sampling, round ',self.rnd,'...')
                self.exploit(self.model)
        
            self.rnd += 1


        #should only reach this stage if we are doing data_sampling
        #  
        for self.rnd in range (self.rnd,self.Iterations+1):
            #First predict data  
            self.step == 'Exploitative'
            print('Exploitative Sampling, round ',self.rnd,'...')
            self.exploit(self.model)

            self.step = 'Explorative'
            print('Begin Explorative Sampling, round ',self.rnd,'...')
            self.explore()

            self.step = 'Sampling'
            self.sample_data(self.rnd)

            
            
            #next Training
            self.step == 'Model_training'
            if self.rnd == 1 or not self.better_than_prev(self.rnd-1):
                print('Perform hyperparameter search...')
                self.hyperparameter_search(self.rnd)
            else:
                print('Train surrogate model, round ',self.rnd,'...')
                self.train()
