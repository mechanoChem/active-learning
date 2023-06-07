import os, sys
import numpy as np
from datetime import datetime
from shutil import copyfile
from tensorflow import keras
from active_learning.workflow.dictionary import Dictionary
from active_learning.workflow.hp_search import hyperparameterSearch
from active_learning.data_recommended.DataRecommender import DataRecommender
from active_learning.data_collector.Sampling import Sampling


class Workflow():

    def __init__(self,input_path):
        self.dict = Dictionary(input_path)
        if self.dict.verify_necessary_information == False:
            print('Input File does not have necessary information')
            exit()
        [self.Model_type,    
         self.Data_Generation, self.Data_Generation_Source, 
         self.restart, self.Restart_path, self.Input_data, self.Input_dim, 
         self.Output_Dim, self.Derivative_Dim, self.Iterations, self.OutputFolder, 
         self.seed, self.temperatures] = self.dict.get_category_values('Overview')
        
        if self.restart=='True':
            self.read_restart(self.Restart_path)
        else:
            self.rnd=0
            if  os.path.exists(self.OutputFolder + 'training'):
                os.rmdir(self.OutputFolder +'training')
            if  os.path.exists(self.OutputFolder +'data_recommended'):
                os.rmdir(self.OutputFolder +'data_recommended')
            if  os.path.exists(self.OutputFolder +'data_sampled'):
                os.rmdir(self.OutputFolder +'data_sampled')
            if  os.path.exists(self.OutputFolder +'outputFiles'):
                os.rmdir(self.OutputFolder +'outputFiles')
            if not os.path.exists(self.OutputFolder):
                os.mkdir(self.OutputFolder)
            os.mkdir(self.OutputFolder +'training')
            os.mkdir(self.OutputFolder +'data_recommended')
            os.mkdir(self.OutputFolder +'data_sampled')
            os.mkdir(self.OutputFolder +'outputFiles')
            if self.seed == '':
                self.seed = 1
            self.construct_model()
        if self.Input_data != 'False':
            self.sampling = Sampling(self.model, self.dict)
            #self.step=1 is explorative sampling, self.step=2 is model training, self.step=3 is hyperparameter search 
            self.step = 'Model_training'
            self.handle_input_data(self.Input_data)
        else:
            self.step = 'Explorative'
        self.recommender = DataRecommender(self.model,self.dict)


    def construct_model(self):
        if self.Model_type == 'IDNN':
            from active_learning.model.idnn_model import IDNN_Model 
            self.model = IDNN_Model(self.dict)



    def read_restart(self,input_path):
         self.dict_restart = Dictionary(input_path)
         [self.rnd, self.step, self.input_data, self.model] = self.dict_restart.get_category_values('Restart')

    def save_restart(self, restart_path):
        return True 


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
        [self.sample_wells,self.sample_vertices,self.test_set] = self.dict.get_individual_keys(['sample_wells','sample_vertices','test_set'])
        for t in self.temperatures:
            if 'Explore' in self.sample_wells:
                self.recommender.sample_wells(self.rnd,t)
            if self.sample_vertices != 'False':
                self.recommender.sample_vertices(self.rnd,t)
            if self.test_set == 'sobol':
                self.recommender.explore(self.rnd,self.test_set, x_bounds=[self.xmin, self.xmax], temp =t)
            else:
                self.recommender.explore(self.rnd,self.test_set, temp =t)

    
    def exploit(self,model,rnd):
        for t in self.temperatures:
            if self.high_error == True:
                self.recommender.high_error(self.rnd,t)
            if self.hessian == True:
                self.recommender.hessian(self.rnd,t)
            if 'Exploit' in self.sample_wells:
                self.recommender.find_wells(self.rnd,t)



    def sample_data(self,rnd):
        self.sampling.read(rnd)


    def hyperparaemter_search(self,rnd):

        # submit
        commands = [f"sys.path.append('{os.path.dirname(__file__)}')",
                    'from active_learning import Active_learning']
        training_func = 'Active_learning("{}").train_rand_idnn'.format(self.config_path)
    
        self.hidden_units, self.lr = hyperparameterSearch(rnd,self.N_hp_sets,commands,training_func, self.job_manager,self.Account,self.Walltime,self.Mem,)

    
    def IDNN_transforms(self):

        sys.path.append(self.config_path)
        from TransformsModule import transforms 

        return transforms

    def better_than_prev(self,rnd):

        [params] = self.data_sampling.read(rnd,only_recent=False)
     
        # Get loss of current idnn
        current_loss = self.model.loss(params)

        # Get loss of previous idnn
        keras.backend.clear_session()
        lastmodel = self.model.load_model(rnd-1, custom_objects={'Transform': Transform(self.IDNN_transforms())})
        prev_loss = self.model.loss(params)

        # Reload current IDNN
        
        if current_loss < prev_loss:
            return True
        else:
            return False
    
    def train(self):
        self.model.train(self.rnd)
        
        
        


    def main_workflow(self):

        #If there is no pre-existing data : we first have to do explorative predictions 
        if self.step == 'Explorative':
            print('Explorative Data Recommendations, round ',self.rnd,'...')
            self.explore()
            if self.Data_Generation==True:
                print('Data sampling, round ',self.rnd,'...')
                self.sample_data(self.rnd)
                self.step == 'Model_training'
            else:
                self.step = 'Complete'

        if self.step == 'Model_training':
            print('Train surrogate model, round ',self.rnd,'...')
            self.model = self.train()
            if self.Data_Generation==False:
                self.model = self.hyperparameter_search(self.rnd)
                self.step = 'Complete'
                print('Exploitative Sampling, round ',self.rnd,'...')
                self.exploit(self.model,self.rnd)
        
        self.rnd += 1

        #should only reach this stage if we are doing data_sampling
        #  
        for self.rnd in range (self.rnd+1,self.Iterations):
            #First predict data  
            self.step == 'Exploitative'
            print('Exploitative Sampling, round ',self.rnd,'...')
            self.exploit(self.model,self.rnd)

            self.step = 'Explorative'
            print('Begin Explorative Sampling, round ',self.rnd,'...')
            self.explore(self.rnd)

            self.step = 'Sampling'
            self.sample_data(self.rnd)

            
            
            #next Training
            self.step == 'Model_training'
            if self.rnd == 1 or not self.better_than_prev(self.model):
                print('Perform hyperparameter search...')
                self.model = self.hyperparameter_search(self.rnd)
            else:
                print('Train surrogate model, round ',self.rnd,'...')
                self.model = self.train(self.model,self.rnd)
