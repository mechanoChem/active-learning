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
from active_learning.workflow.make_graph import graph

class Workflow():

    def __init__(self,input_path):
        

        self.input_path = input_path
        self.dict = Dictionary(input_path)
        # print(self.dict.get_category_values('Overview'))
        [self.Model_type,    
         self.Data_Generation, self.Data_Generation_Source, 
         self.restart, self.Input_data, self.Input_alias, 
         self.Output_alias,self.Iterations, self.OutputFolder, 
         self.seed,self.input_dim,self.output_dim,self.derivative_dim,self.config_path,self.T, self.graph,self.reweight,self.reweight_alpha,self.prediction_points] = self.dict.get_category_values('Main')
        
        
        [self.N_global_pts, self.sample_external,self.external_path,
         self.external_points,self.external_perturbation] = self.dict.get_category_values('Explore_Parameters')



        [self.sample_non_convexities,self.sample_high_error,
         self.find_wells,self.lowest_free_energy, 
         self.sample_sensitivity,self.QBC] = self.dict.get_individual_keys('Exploit_Parameters',['non_convexities','high_error','find_wells','lowest_free_energy','sensitivity','qbc'])
        if self.restart==False:
            self.rnd=0
            if  os.path.exists(self.OutputFolder + 'training'):
                shutil.rmtree(self.OutputFolder +'training')
            if  os.path.exists(self.OutputFolder + 'graphs'):
                shutil.rmtree(self.OutputFolder +'graphs')
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
            os.mkdir(self.OutputFolder +'graphs')
            os.mkdir(self.OutputFolder +'data/data_recommended')
            os.mkdir(self.OutputFolder +'data/data_sampled')
            os.mkdir(self.OutputFolder +'data/outputFiles')
            
        # self.rnd=0
        # self.construct_model()
        # self.sampling = CASM_Sampling(self.model, self.dict)
        # self.sample_data(self.rnd)
        
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
        self.recommender.construct_input_types()
        self.first_explore=True
        self.first_exploit=True
        self.stopping_criteria_met=False

        # self.recommender.query_by_committeee(1,2)


    def construct_model(self):
        if self.Model_type == 'IDNN':
            [self.transform_path] = self.dict.get_individual_keys('IDNN',['transforms_directory'])
            from active_learning.model.idnn_model import IDNN_Model 
            self.model = IDNN_Model(self.dict)



    def read_restart(self):
        #  self.dict_restart = Dictionary(input_path)
         [self.rnd, self.step] = self.dict.get_category_values('Restart')
        #  self.model.load_trained_model(self.rnd-1)

    # def save_restart(self):
    #     params = [self.rnd, self.step]
    #     jsonparams = json.dumps(params)
    #     with open(self.outputFolder + 'restart.json', "w") as outfile:
    #         outfile.write(jsonparams)


    def handle_input_data(self,input_file):
        input_results = np.loadtxt('allResults49.txt')
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
        if self.first_explore:
            self.recommender.create_types("billiardwalk")
            if self.sample_external:
                self.recommender.create_types('Externally_supplied_points')
            self.first_explore=False
        self.recommender.explore(self.rnd)
        if self.sample_external:
            self.recommender.sample_external_data(self.rnd)



    
    def exploit(self,model):
        if self.first_exploit:
            if self.sample_high_error:
                self.recommender.create_types("high_error")
            if self.find_wells:
                self.recommender.create_types('find_wells')
            if self.find_wells:
                self.recommender.create_types('sample_vertices')
            if self.lowest_free_energy:
                self.recommender.create_types('lowest_free_energy')
            if self.sample_non_convexities:
                self.recommender.create_types('non_convexities')
            if self.sample_sensitivity:
                self.recommender.create_types('sensitivity')
            self.first_exploit=False

        self.recommender.get_latest_pred(self.rnd)
        if self.sample_non_convexities or self.sample_sensitivity:
            self.recommender.explore_extended(self.rnd)
            self.recommender.predict_explore_extended(self.rnd)
            self.recommender.find_eigenvalues_explore(self.rnd)
        if self.sample_high_error == True:
            self.recommender.high_error(self.rnd)
        if self.sample_non_convexities == True:
            self.recommender.non_convexities(self.rnd)
        if self.find_wells:
            self.recommender.find_wells(self.rnd)
        if self.lowest_free_energy:
            self.recommender.lowest_free_energy_curve(self.rnd)
        if self.sample_sensitivity:
            self.recommender.sensitivity(self.rnd)




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
            commands = [f"sys.path.append('{sys.path[0]}')",
                    'from active_learning.model.idnn_model import IDNN_Model',
                    'from active_learning.workflow.dictionary import Dictionary',
                     f'dict = Dictionary("{os.path.abspath(self.input_path)}")',
                     'model  = IDNN_Model(dict)']
        training_func = 'model.train_rand_idnn'.format(self.config_path)

        if not self.sample_non_convexities and not self.sample_sensitivity:
            self.recommender.explore_extended(rnd)
    
        params = hyperparameterSearch(rnd,N_hp_sets,commands,training_func, job_manager,account,walltime,mem,self.OutputFolder)
        self.model.load_trained_model(rnd)
        if self.QBC:
            self.recommender.query_by_committeee(rnd,N_hp_sets)
    
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
        lastmodel = self.model.load_trained_model(rnd-1)#, custom_objects={'Transform': self.IDNN_transforms()})
        prev_loss = self.model.loss(rnd)


        # Reload current IDNN
        self.model.load_trained_model(rnd)#, custom_objects={'Transform': self.IDNN_transforms()})
        if current_loss < prev_loss:
            return True
        else:
            return False
    
    def train(self):
        self.model.train(self.rnd)
        self.model.save_model(self.rnd)


    def main_workflow(self):

        # self.train()

        # self.model = self.hyperparameter_search(self.rnd)

        # If there is no pre-existing data : we first have to do explorative predictions
        # self.hyperparameter_search(2) 
        # self.better_than_prev(1)
        if self.step == 'Explorative':
            print('Explorative Data Recommendations, round ',self.rnd,'...')
            self.explore()
            self.recommender.choose_points(self.rnd)
            if self.Data_Generation==True:
                self.step = 'Sampling'
            else:
                self.step = 'Complete'
        
        # print(self.step)
        # self.step='Sampling'

        if self.step == 'Sampling':
            print('Data sampling, round ',self.rnd,'...')
            self.sample_data(self.rnd)
            self.step = 'Model_training'


        # self.step ='Model_training'
        

        if self.step == 'Model_training':
            print('Train surrogate model, round ',self.rnd,'...')
            self.train()
            if self.Data_Generation==False:
                self.model = self.hyperparameter_search(self.rnd)
                self.step = 'Complete'
                # self.rnd += 1
                print('Exploitative Sampling, round ',self.rnd,'...')
                self.exploit(self.model)
                self.recommender.choose_points(self.rnd)
            elif self.rnd == 1:
                self.model = self.hyperparameter_search(self.rnd)
            graph(self.rnd, self.model,self.dict)
        
            self.rnd += 1


        #should only reach this stage if we are doing data_sampling
        #  
        for self.rnd in range (self.rnd,self.Iterations+1):
            #First predict data  



            if self.rnd>1:
                improvements = self.recommender.determine_improvement(self.rnd)
                if self.recommender.stopping_criteria(improvements):
                    print('Stopping Criteria Met for rnd', self.rnd)
                    if self.stopping_criteria_met:
                        print('Stopping criteria Met 2 rounds in a row \n Ending AL')
                        sys.exit()
                    else:
                        self.stopping_criteria_met=True

                else:
                    self.stopping_criteria_met=False
            
                if self.reweight:
                    self.recommender.reweight_criterion(self.rnd,improvements)

            self.step == 'Exploitative'
            print('Exploitative Sampling, round ',self.rnd,'...')
            self.exploit(self.model)

            self.step = 'Explorative'
            print('Begin Explorative Sampling, round ',self.rnd,'...')
            self.explore()

            self.recommender.choose_points(self.rnd)

            self.step = 'Sampling'
            self.sample_data(self.rnd)

            
            
            # next Training
            self.step == 'Model_training'
            # if 1 == 0:#
            if self.rnd == 1 or not self.better_than_prev(self.rnd-1):
                # print('Train surrogate model, round ',self.rnd,'...')
                # self.train()
                print('Perform hyperparameter search...')
                if self.rnd==1:
                    self.recommender.create_types('QBC')
                    
                self.hyperparameter_search(self.rnd)
                

            else:
                print('Train surrogate model, round ',self.rnd,'...')
                self.train()

            graph(self.rnd, self.model,self.dict)
