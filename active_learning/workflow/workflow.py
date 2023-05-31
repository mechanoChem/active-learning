import os, sys
import dictionary
import numpy as np
from datetime import datetime
from shutil import copyfile
from tensorflow import keras



class Workflow():

    def __init__(self,input_path):
        self.dict = dictionary(input_path)
        if self.dict.verify_necessary_information == False:
            print('Input File does not have necessary information')
            exit()
        [self.Model, self.Data_Generation, self.Data_Generation_Source, 
         self.Restart, self.Restart_path self.Input_data, self.Input_dim, 
         self.Output_Dim, self.Derivative_Dim, self.Iterations, self.Ouput_folder, 
         self.seed, self.temperatures] = self.dict.get_category_values('Overview')
        if self.restart:
                self.read_restart(self.Restart_path)
        else:
            self.rnd=0
            if  os.path.exists(self.OutputFolder + 'training'):
                os.rmdir(self.OutputFolder +'training')
            if  os.path.exists(self.OutputFolder +'data_recommended'):
                os.rmdir(self.OutputFolder +'data_recommend')
            if  os.path.exists(self.OutputFolder +'data_sampled'):
                os.rmdir(self.OutputFolder +'data_sampled')
            if  os.path.exists(self.OutputFolder +'outputFiles'):
                os.rmdir(self.OutputFolder +'outputFiles')
            os.mkdir(self.OutputFolder +'training')
            os.mkdir(self.OutputFolder +'data_recommended')
            os.mkdir(self.OutputFolder +'data_sampled')
            os.mkdir(self.OutputFolder +'outputFiles')
            if self.seed == '':
                self.seed = 1
            self.construct_model
        if self.Input_data != False:
            #self.step=1 is explorative sampling, self.step=2 is model training, self.step=3 is hyperparameter search 
            self.step = 'Model_training'
        else:
            self.step = 'Explorative'
        if self.input_file != '':
            self.andle_input_data(self.input_file)
        self.recommender = DataRecommender()


    def construct_model(self):
        if self.model == 'IDNN':
            from idnn import IDNN 
            [self.hidden_units, self.activation, self.IDNN_transforms, self.Dropout, 
             self.final_bias] = self.dict.get_category_values('Neural Network')
            self.model = IDNN(self.dim,
                    self.hidden_units,
                    activation = self.activation,
                    transforms=self.IDNN_transforms(),
                    dropout=self.Dropout,
                    unique_inputs=True,
                    final_bias=True)
            self.opt = 'keras.optimizers.' + self.optimizer 
            self.model.compile(loss=['mse','mse',None],
                            loss_weights=[0.01,1,None],
                            optimizer=eval(self.opt)(learning_rate=self.lr))



    def read_restart(self,input_path):
         self.dict_restart = dictionary(input_path)
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




    def explore(self, rnd):
        for t in self.temperatures:
            if self.sample_wells != False:
                self.recommender.sample_wells(rnd,t)
            if self.sample_vertices != False:
                self.recommender.sample_vertices(rnd,t)
            if self.test_set == 'sobol':
                self.recommender.explore(rnd,self.test_set, x_bounds=[self.xmin, self.xmax], temp =t)
            else:
                self.recommender.explore(rnd,self.test_set, temp =t)

    
    def exploit(self,model,rnd):
        for t in self.temperatures:
            if self.high_error == True:
                


    def sample_data(self,rnd):

    def hyperparaemter_search(self,rnd):
    
    def model_bad(self,rnd):
    
    def train(self,model,rnd):
        
        
        


    def main_worflow(self):

        #If there is no pre-existing data : we first have to do explorative predictions 
        if self.step == 'Explorative':
            print('Explorative Data Recommendations, round ',rnd,'...')
            self.explore(self.model,rnd)
            if self.Data_Generation==True:
                print('Data sampling, round ',rnd,'...')
                self.sample_data(rnd)
                self.step == 'Model_training'
            else:
                self.step = 'Complete'

        if self.step == 'Model_training':
            print('Train surrogate model, round ',rnd,'...')
            self.model = self.train(self.model,rnd)
            if self.Data_Generation==False:
                self.model = self.hyperparameter_search(rnd)
                self.step = 'Complete'
                print('Exploitative Sampling, round ',rnd,'...')
                self.exploit(self.model,rnd)
        
        rnd += 1

        #should only reach this stage if we are doing data_sampling
        #  
        for rnd in range (self.rnd+1,self.Iterations):
            #First predict data  
            self.step == 'Exploitative'
            print('Exploitative Sampling, round ',rnd,'...')
            self.exploit(self.model,rnd)

            self.step = 'Explorative'
            print('Begin Explorative Sampling, round ',rnd,'...')
            self.explore(rnd)

            self.step = 'Sampling'
            self.sample_data(rnd)

            
            
            #next Training
            self.step == 'Model_training'
            if rnd == 1 or self.model_bad(self.model):
                print('Perform hyperparameter search...')
                self.model = self.hyperparameter_search(rnd)
            else:
                print('Train surrogate model, round ',rnd,'...')
                self.model = self.train(self.model,rnd)
