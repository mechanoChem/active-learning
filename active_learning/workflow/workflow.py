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
from active_learning.workflow.make_graph import graph, pred_training_points
from active_learning.model.idnn_model import IDNN_Model 
from configparser import ConfigParser
from active_learning.workflow.finalchecks import loss,plotting_points, predicting_best, predicting_all
import time

class Workflow():

    def __init__(self,input_path,only_initialize=False,originalpath=False,ExistingGlobal=True):
        self.ExistingGlobal=ExistingGlobal
        self.start_time = time.time()
        self.last_time = time.time()
        self.pointcount=0

        config = ConfigParser()
        config.read(input_path)
        self.originalpath = os.path.dirname(input_path)
        if originalpath:
            self.originalpath=originalpath
        self.OutputFolder = config['Main']['OutputFolder']
        self.restart = config['Main']['restart']

        print('AL beginning for ',self.OutputFolder)

        if self.restart=="False" and not only_initialize:
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
            if not os.path.exists(self.OutputFolder+'slurm'):
                os.mkdir(self.OutputFolder+'slurm')
            os.mkdir(self.OutputFolder +'training')
            os.mkdir(self.OutputFolder +'training/trainings')
            os.mkdir(self.OutputFolder +'training/predictions')
            os.mkdir(self.OutputFolder +'graphs')
            os.mkdir(self.OutputFolder +'data/data_recommended')
            os.mkdir(self.OutputFolder +'data/data_sampled')
            os.mkdir(self.OutputFolder +'data/outputFiles')
        



        if not only_initialize:
            shutil.copyfile(input_path,self.OutputFolder+'input.ini')




        self.input_path = self.OutputFolder+'input.ini'
        self.dict = Dictionary(self.input_path,self.originalpath)
        [self.Model_type,    
         self.Data_Generation, self.Data_Generation_Source, 
         self.restart, self.Input_data, self.Input_alias, 
         self.Output_alias,self.Iterations, self.OutputFolder, 
         self.seed,self.input_dim,self.output_dim,self.derivative_dim,self.config_path,self.T, self.graph,self.reweight,self.reweight_alpha,self.prediction_points] = self.dict.get_category_values('Main')
        

        
        [self.N_global_pts, self.sample_external,self.external_path,
         self.external_points,self.external_perturbation] = self.dict.get_category_values('Explore_Parameters')

        [self.hyperparameter_search_on] = self.dict.get_individual_keys('IDNN',['idnn_hyperparameter'])


        [self.sample_non_convexities,self.sample_high_error,
         self.find_wells,self.lowest_free_energy, 
         self.sample_sensitivity,self.QBC] = self.dict.get_individual_keys('Exploit_Parameters',['non_convexities','high_error','find_wells','lowest_free_energy','sensitivity','qbc'])
    
            
        if self.Input_data:
            self.step = 'Model_training'
            self.handle_input_data(self.Input_data)
        else:
            self.step = 'Explorative'
        self.construct_model()
        self.first_explore=True
        self.first_exploit=True
            
        if self.Data_Generation:
            if self.Data_Generation_Source=='CASM' or self.Data_Generation_Source=='CASM_Surrogate':
                self.sampling = CASM_Sampling(self.model, self.dict)
        self.recommender = DataRecommender(self.model,self.dict)
        self.recommender.construct_input_types()
        self.stopping_criteria_met=False
        self.hp_search=False
        self.points_2D = self.recommender.points2D()
        if self.restart==True and not only_initialize:
            print('read_restart')
            self.read_restart()
        # self.recommender.query_by_committeee(1,50)

        # self.recommender.query_by_committeee(1,2)


    def construct_model(self):
        if self.Model_type == 'IDNN':
            [self.transform_path] = self.dict.get_individual_keys('IDNN',['transforms_directory'])
            from active_learning.model.idnn_model import IDNN_Model 
            self.model = IDNN_Model(self.dict)



    def read_restart(self):
        #  self.dict_restart = Dictionary(input_path)
        [self.rnd, self.step] = self.dict.get_category_values('Restart')
        self.first_exploit=False
        self.first_explore=False
        self.recommender.return_weights_default()
        self.pointcount=self.N_global_pts*self.rnd
        if self.step == "Sampling_complete":
            if (not self.onlyGlobal()) or (not self.ExistingGlobal):
                output = self.sampling.read_from_casm(self.rnd,self.ExistingGlobal,self.pointcount,self.pointcount+self.N_global_pts,self.recommender.explore2d(),self.points_2D)
                self.recommender.append(self.rnd,output)
            else:
                if self.ExistingGlobal:
                    if self.recommender.explore2d():
                        billardpoints = np.genfromtxt('shuffled.txt',dtype=np.float64)[round(self.pointcount*(1-self.points_2D)):round(self.pointcount*(1-self.points_2D))+round(self.N_global_pts*(1-self.points_2D)),:]
                        self.save(billardpoints,rnd,'billiardwalk_preexisting')
                        # print('range for billiard2D',round(self.pointcount*self.points_2D),'to',round(self.pointcount*self.points_2D)+round(self.N_global_pts*self.points_2D))
                        billardpoints2D = np.genfromtxt('billiard2D.txt',dtype=np.float64)[round(self.pointcount*self.points_2D):round(self.pointcount*self.points_2D)+round(self.N_global_pts*self.points_2D),:]
                        self.save(billardpoints2D,rnd,'billiardwalk2D_preexisting')
                        billardpoints = np.vstack((billardpoints,billardpoints2D))
                    else:
                        billardpoints = np.genfromtxt('shuffled.txt',dtype=np.float64)[self.pointcount:self.pointcount+self.N_global_pts,:]
                        self.save(billardpoints,rnd,'billiardwalk_preexisting')
                    kappa = billardpoints[:,:self.derivative_dim]
                    eta = billardpoints[:,self.derivative_dim:2*self.derivative_dim]
                    phi= billardpoints[:,2*self.derivative_dim:3*self.derivative_dim]
                    T = billardpoints[:,-self.derivative_dim-1:-self.derivative_dim]
                    mu =billardpoints[:,-self.derivative_dim:] 
                    output = np.hstack((kappa,T, np.zeros(np.shape(T))))
                    self.recommender.append(self.rnd,output)
                    self.sampling.write(self.rnd,kappa,eta,phi,T,mu)
            self.step='Model_training'
        if self.step == "HPcomplete":
            current_loss=1000
            hp_loss = self.hyperparameter_search(self.rnd,current_loss,readonly=True)
            print('hp_loss',hp_loss)
            self.step ="Exploitative"
            self.rnd+= 1
        # print('self.rnd',self.rnd)
        if self.reweight and self.rnd>1:
            improvements =self.recommender.determine_improvement(self.rnd-1)
            self.recommender.reweight_criterion(self.rnd-1,improvements)
            

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
            # if not self.ExistingGlobal:
            self.recommender.create_types("billiardwalk")
            if self.recommender.explore2d():
                self.recommender.create_types("billiardwalk_2d")
            if self.sample_external:
                self.recommender.create_types("sample_wells")
                self.recommender.create_types("sample_vertices")
                # self.recommender.create_types('Externally_supplied_points')
            self.first_explore=False
        if not self.ExistingGlobal:
            self.recommender.explore(self.rnd)
            if self.recommender.explore2d():
                self.recommender.explore(self.rnd,twoD=True)
        if self.sample_external:
            self.recommender.sample_wells(self.rnd)
            self.recommender.sample_vertices(self.rnd)
            # self.recommender.sample_external_data(self.rnd)


    
    def exploit(self,model):
        if self.first_exploit:
            if self.sample_high_error:
                self.recommender.create_types("high_error")
            if self.find_wells:
                self.recommender.create_types('find_wells')
            if self.lowest_free_energy:
                self.recommender.create_types('lowest_free_energy')
            if self.sample_non_convexities:
                self.recommender.create_types('non_convexities')
            if self.sample_sensitivity:
                self.recommender.create_types('sensitivity')
            self.first_exploit=False

        # self.recommender.get_latest_pred(self.rnd)
        if self.sample_non_convexities or self.sample_sensitivity or self.find_wells:
            self.recommender.explore_extended(self.rnd,twoD=(self.recommender.explore2d()))
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


    def onlyGlobal(self):
        if len(self.recommender.get_types())> 2:
            return False
        return True


    def save(self,output,rnd,type):
        np.savetxt(self.OutputFolder+'data/data_recommended/'+type+'_rnd'+str(rnd)+'.txt',output,fmt='%.12f')

    def sample_data(self,rnd):
        if (not self.onlyGlobal() or not self.ExistingGlobal) and not (rnd==0 and self.ExistingGlobal and self.sample_external==False):
            command = self.sampling.construct_job(rnd)
            self.sampling.submit_job(command)
            output =self.sampling.read_from_casm(rnd,self.ExistingGlobal,self.pointcount,self.pointcount+self.N_global_pts,self.recommender.explore2d(),self.points_2D)
            self.recommender.append(rnd,output)
            # if self.ExistingGlobal:
            #     if self.recommender.explore2d:
            #         billardpoints = np.genfromtxt('shuffled.txt',dtype=np.float64)[round(self.pointcount*(1-self.points_2D)):round(self.pointcount*(1-self.points_2D))+round(self.N_global_pts*(1-self.points_2D)),:]
            #         self.save(billardpoints,rnd,'billiardwalk_preexisting')
            #         # print('range for billiard2D',round(self.pointcount*self.points_2D),'to',round(self.pointcount*self.points_2D)+round(self.N_global_pts*self.points_2D))
            #         billardpoints2D = np.genfromtxt('billiard2D.txt',dtype=np.float64)[round(self.pointcount*self.points_2D):round(self.pointcount*self.points_2D)+round(self.N_global_pts*self.points_2D),:]
            #         self.save(billardpoints2D,rnd,'billiardwalk2D_preexisting')
            #         billardpoints = np.vstack((billardpoints,billardpoints2D))
            #     else:
            #         billardpoints = np.genfromtxt('shuffled.txt',dtype=np.float64)[self.pointcount:self.pointcount+self.N_global_pts,:]
            #         self.save(billardpoints,rnd,'billiardwalk_preexisting')
            #     kappa = billardpoints[:,:self.derivative_dim]
            #     eta = billardpoints[:,self.derivative_dim:2*self.derivative_dim]
            #     phi= billardpoints[:,2*self.derivative_dim:3*self.derivative_dim]
            #     T = billardpoints[:,-self.derivative_dim-1:-self.derivative_dim]
            #     mu =billardpoints[:,-self.derivative_dim:] 
            #     output = np.hstack((eta,T, np.zeros(np.shape(T))))
            #     self.recommender.append(rnd,output)
        else:
            if self.recommender.explore2d():
                billardpoints = np.genfromtxt('shuffled.txt',dtype=np.float64)[round(self.pointcount*(1-self.points_2D)):round(self.pointcount*(1-self.points_2D))+round(self.N_global_pts*(1-self.points_2D)),:]
                self.save(billardpoints,rnd,'billiardwalk_preexisting')
                # print('range for billiard2D',round(self.pointcount*self.points_2D),'to',round(self.pointcount*self.points_2D)+round(self.N_global_pts*self.points_2D))
                billardpoints2D = np.genfromtxt('billiard2D.txt',dtype=np.float64)[round(self.pointcount*self.points_2D):round(self.pointcount*self.points_2D)+round(self.N_global_pts*self.points_2D),:]
                self.save(billardpoints2D,rnd,'billiardwalk2D_preexisting')
                billardpoints = np.vstack((billardpoints,billardpoints2D))
            else:
                billardpoints = np.genfromtxt('shuffled.txt',dtype=np.float64)[self.pointcount:self.pointcount+self.N_global_pts,:]
                self.save(billardpoints,rnd,'billiardwalk_preexisting')
            kappa = billardpoints[:,:self.derivative_dim]
            eta = billardpoints[:,self.derivative_dim:2*self.derivative_dim]
            phi= billardpoints[:,2*self.derivative_dim:3*self.derivative_dim]
            T = billardpoints[:,-self.derivative_dim-1:-self.derivative_dim]
            mu =billardpoints[:,-self.derivative_dim:] 
            output = np.hstack((kappa,T, np.zeros(np.shape(T))))
            self.recommender.append(rnd,output,onlyInitial=True)
            print('choose billiard points')
            self.recommender.choose_points(rnd)
            print('end of choose points')
            kappas_kept =self.recommender.read_recommended(rnd)[:,0:7]
            print("shape kappas kept",np.shape(kappas_kept))
            # I = np.zeros((kappa.shape[0]), dtype=int)
            # # Loop through each row of kappa and check if it's in kappas_kept
            # for i in range(kappa.shape[0]):
            #     if np.any(np.all(kappa[i]- kappas_kept < 1e-12, axis=1)):
            #         I[i]=1

            # I = I[I==1]
            # print('kappa',np.shape(kappa))
            # print('kappaI-12',np.shape(kappa[I,:]))
            
            
            I = np.zeros(kappa.shape[0], dtype=bool) 
            # Loop through each row of kappa and check if it's in kappas_kept
            for i in range(kappa.shape[0]):
                if np.any(np.all(kappa[i]- kappas_kept < 1e-11, axis=1)):
                    I[i] = True

            # I = [I==1]
            # print('kappa',np.shape(kappa))
            # print('kappaI-11',np.shape(kappa[I,:]))
            # print('I',I)
            # print(kappa)
            # print(kappa[I,:])
            
            self.sampling.write(rnd,kappa[I,:],eta[I,:],phi[I,:],T[I,:],mu[I,:])
        
        self.pointcount+=self.N_global_pts


    def hyperparameter_search(self,rnd,loss,original=True,readonly=False):

        [N_hp_sets] = self.dict.get_individual_keys('IDNN_Hyperparameter',['n_sets'])
        job_manager,account,walltime,mem, = self.dict.get_category_values('Hyperparameter_Job_Manager')
        # submit
        if self.Model_type=='IDNN':
            commands = [f"sys.path.append('{sys.path[0]}')",
                    'from active_learning.model.idnn_model import IDNN_Model',
                    'from active_learning.workflow.dictionary import Dictionary',
                     f'dict = Dictionary("{self.input_path}","{self.originalpath}")',
                     'model  = IDNN_Model(dict)']
        training_func = 'model.train_rand_idnn'.format(self.config_path)

        if not self.sample_non_convexities and not self.sample_sensitivity:
            self.recommender.explore_extended(rnd,twoD=self.recommender.explore2d())
    
        params,loss = hyperparameterSearch(rnd,N_hp_sets,commands,training_func, job_manager,account,walltime,mem,self.OutputFolder,loss,original,readonly)
        self.model.load_trained_model(rnd)
        if self.QBC:
            self.recommender.query_by_committeee(rnd,N_hp_sets)
        self.recommender.return_weights_default()
        return loss

    def IDNN_transforms(self):


        sys.path.append(self.config_path+self.transform_path)
        from TransformsModule import transforms 

        return transforms

    def better_than_prev(self,rnd):


     
        # Get loss of current idnn

        #given data for rnd r, evaluates loss for rnd r-1
        current_loss = self.model.loss(rnd+1)
        print('current loss',current_loss)

        # Get loss of previous idnn
        keras.backend.clear_session()
        #given data for rnd r, evaluates loss for rnd r-2
        lastmodel = self.model.load_trained_model(rnd-1)#, custom_objects={'Transform': self.IDNN_transforms()})
        prev_loss = self.model.loss(rnd+1)

        print('prev loss',prev_loss)


        # Reload current IDNN
        self.model.load_trained_model(rnd)#, custom_objects={'Transform': self.IDNN_transforms()})
        if current_loss < prev_loss:
            return True
        else:
            return False
    
    def train(self):
        loss =self.model.train(self.rnd)
        self.model.save_model(self.rnd)
        return loss
    
    def finalize(self,rnd,findloss=True):
        if findloss:
            loss(self.dict,self.model,self.OutputFolder,rnd)
        else:
            predicting_best(self.dict,self.model,self.OutputFolder,rnd)
        plotting_points(self.dict,self.model,self.OutputFolder,rnd)

    def predict_all(self,rnd):
        predicting_all(self.dict,self.model,self.OutputFolder,rnd)

    def make_graphs(self,rnd):
        self.model.load_trained_model(rnd)
        # graph(rnd, self.model,self.dict)
        pred_training_points(rnd, self.model,self.dict)

    def main_workflow(self):
        if self.step == 'Explorative':
            print('Explorative Data Recommendations, round ',self.rnd,'...')

            self.explore()
            self.recommender.choose_points(self.rnd)
            if self.Data_Generation==True:
                self.step = 'Sampling'
            else:
                self.step = 'Complete'
        

        if self.step == 'Sampling':
            print('Data sampling, round ',self.rnd,'...')

            self.sample_data(self.rnd)
            self.step = 'Model_training'


        # self.step ='Model_training'
        

        if self.step == 'Model_training':
            print('Train surrogate model, round ',self.rnd,'...')

            self.train()
            print('Training Complete')

            if self.Data_Generation==False:
                self.model = self.hyperparameter_search(self.rnd)
                self.step = 'Complete'
                # self.rnd += 1
                print('Exploitative Sampling, round ',self.rnd,'...')
                self.exploit(self.model)
                current_time = time.time()
                # print('Current time',current_time -self.start_time)
                # print('Time since last time',current_time -self.last_time)
                self.last_time = current_time 
                self.recommender.choose_points(self.rnd)
            elif self.rnd == 1:
                self.hyperparameter_search(self.rnd,1000)
            graph(self.rnd, self.model,self.dict)
            pred_training_points(self.rnd, self.model,self.dict)
        
            self.rnd += 1


        #should only reach this stage if we are doing data_sampling
        #  
        for self.rnd in range (self.rnd,self.Iterations+1):
            #First predict data  



            if self.rnd>1:
                if self.reweight:
                    improvements =self.recommender.determine_improvement(self.rnd-1)
                    self.recommender.reweight_criterion(self.rnd-1,improvements)

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
            if self.hyperparameter_search_on and (self.rnd == 1 or not self.better_than_prev(self.rnd-1)):
                self.hp_search=True
                print('Perform hyperparameter search...')
                
                if self.rnd==1 and self.QBC:
                    self.recommender.create_types('QBC')
                current_loss = self.train()
                print('Finished getting current loss')
                


                hp_loss = self.hyperparameter_search(self.rnd,current_loss)
                print('Finished hyperparameter search')
                
                print('current_loss',current_loss)
                print('hp_loss',hp_loss)
                # if self.rnd !=1 and current_loss < hp_loss:
                #     print('current_loss',current_loss)
                #     print('hp_loss',hp_loss)
                #     print('HP search did not yield lower mse - trying again')
                #     hp_loss_2 = self.hyperparameter_search(self.rnd,current_loss,original=False)
                #     if current_loss < hp_loss_2:
                #         print('HP search did not yield lower mse \n Ending AL')
                #         sys.exit()
            else:
                self.hp_search=False
                print('Train surrogate model, round ',self.rnd,'...')
                
                self.train()
                print("Finished training")
                

            graph(self.rnd, self.model,self.dict)
            pred_training_points(self.rnd, self.model,self.dict)
            print('Finished graphs')


            # improvements = self.recommender.determine_improvement(self.rnd)
            if self.recommender.stopping_criteria(self.rnd):
                print('Stopping Criteria Met for rnd', self.rnd)
                if self.stopping_criteria_met and self.hp_search:
                    print('Stopping criteria Met 2 rounds in a row \n Ending AL')
                    # sys.exit()
                else:
                    self.stopping_criteria_met=True

            else:
                print('Stopping Criteria Not Met for rnd', self.rnd)
                self.stopping_criteria_met=False

        self.finalize(self.rnd)
