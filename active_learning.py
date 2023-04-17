#!/usr/bin/env python
#combo 

import sys, os

import numpy as np
import shutil
from shutil import copyfile
from idnn import IDNN, find_wells
from mechanoChemML.src.transform_layer import Transform
from mechanoChemML.workflows.active_learning.hp_search import hyperparameterSearch

from importlib import import_module
from data_generation_wrapper import submitCASM, compileCASMOutput, loadCASMOutput
import tensorflow as tf
from sobol_seq import i4_sobol
from mechanoChemML.workflows.active_learning.hitandrun import billiardwalk

from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from configparser import ConfigParser
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda
from numpy import linalg as LA

############ Active learning class #######

class Active_learning(object):
    """
    Class to define the active learning workflow
    used to create a deep neural network
    representation of the free energy of a system.
    """

    ########################################
    
    def __init__(self,config_path):

        if not os.path.exists('training'):
            os.mkdir('training')
        if not os.path.exists('data'):
            os.mkdir('data')
        if not os.path.exists('outputFiles'):
            os.mkdir('outputFiles')

        self.read_config(config_path)
        self.config_path = config_path
        self.seed = 1 # initial Sobol sequence seed
        self.adjustedx = 20000
        self.adjustedn = 500000

        # initialize IDNN
    
        self.idnn = IDNN(self.dim,
                         self.hidden_units,
                         activation = self.activation,
                         transforms=self.IDNN_transforms(),
                         dropout=self.Dropout,
                         unique_inputs=True,
                         final_bias=True)
        self.opt = 'keras.optimizers.' + self.optimizer 
        self.idnn.compile(loss=['mse','mse',None],
                          loss_weights=[0.01,1,None],
                          optimizer=eval(self.opt)(learning_rate=self.lr))

    ########################################

    def read_config(self,config_path):

        config = ConfigParser()
        config.read(config_path)
        folder = os.path.dirname(os.path.abspath(config_path))
        Qpath = folder + '/Q.txt'
        Wpath = folder + '/Wells.txt'
        Vpath = folder + '/Vertices.txt'

        self.casm_project_dir = config['DNS']['CASM_project_dir']
        self.job_manager = config['DNS']['JOB_MANAGER']
        self.CASM_version = config['DNS']['CASM_version']
        self.Account = config['DNS']['Account']
        self.Walltime = config['DNS']['Walltime']
        self.Mem = config['DNS']['Mem']
        self.test = config['DNS']['Test']
        self.N_jobs = int(config['HPC']['CPUNum']) #number of processors to use
        self.N_global_pts = int(config['WORKFLOW']['N_global_pts']) #global sampling points each iteration
        self.N_rnds = int(config['WORKFLOW']['Iterations'])
        self.Epochs = int(config['NN']['Epochs'])
        self.Batch_size = int(config['NN']['Batch_size'])
        self.N_hp_sets = int(config['HYPERPARAMETERS']['N_sets'])
        self.Dropout = float(config['HYPERPARAMETERS']['Dropout'])
        self.activation = [str(p) for p in config['WORKFLOW']['Activation'].split(',')]
        self.hidden_units = [int(p) for p in config['WORKFLOW']['Hidden Units'].split(',')]
        self.test_set = config['WORKFLOW']['Test_set']
        self.dim = config['WORKFLOW']['Dim']
        self.optimizer =  (config['WORKFLOW']['Optimizer'])
        self.LR_range = [float(p) for p in config['HYPERPARAMETERS']['LearningRate'].split(',')]
        self.lr = float(config['HYPERPARAMETERS']['Learning'])
        self.Layers_range = [int(p) for p in config['HYPERPARAMETERS']['Layers'].split(',')]
        self.Neurons_range = [int(p) for p in config['HYPERPARAMETERS']['Neurons'].split(',')]
        self.dim = int(config['WORKFLOW']['Dim'])
        self.Test_set = config['WORKFLOW']['Test_set']
        self.Initial_mu = config['WORKFLOW']['Initial_mu']
        self.Sample_wells = [str(p) for p in  config['WORKFLOW']['Sample_Wells'].split(',')] #Can be any of None, exploit, Guided 
        self.WeightRecent = config['SURROGATE']['WeightRecent']
        self.LR_decay = float(config['SURROGATE']['LR_decay'])
        self.Min_lr = config['SURROGATE']['Min_lr']
        self.EarlyStopping = config['SURROGATE']['EarlyStopping']
        self.Patience = config['SURROGATE']['Patience']

        self.Tmax = float(config['WORKFLOW']['TemperatureMax'])
        self.Tmin = float(config['WORKFLOW']['TemperatureMin'])
        self.phi = np.array([float(p) for p in config['WORKFLOW']['Phi'].split(',')])
        self.Q = np.loadtxt(f'{Qpath}')
        if 'Guided' in self.Sample_wells:
            self.Wells= np.loadtxt(f'{Wpath}')
            self.Vertices = np.loadtxt(f'{Vpath}')
        self.invQ = np.linalg.inv(self.Q)[:,:self.dim]
        self.Q = self.Q[:self.dim]
        self.n_planes = np.vstack((self.invQ,-self.invQ))
        self.c_planes = np.hstack((np.ones(self.invQ.shape[0]),np.zeros(self.invQ.shape[0])))
        self.job_manager = config['DNS']['JOB_MANAGER']
        self.x0 = np.zeros(self.dim)
        self.x0[0] = 0.5
        self.data_generation=[]
        if self.test=='True':
            self.Hidden_Layers = config['DATA_GENERATION']['Hidden_Layers']
            self.data_gen_activation = config['DATA_GENERATION']['Activation']
            self.Input_Shape =  config['DATA_GENERATION']['Input_Shape']
            self.data_generation = [self.Hidden_Layers, self.Input_Shape, self.dim, self.CASM_version, self.data_gen_activation, folder]
        if self.job_manager == 'PC' and self.N_jobs > 1:
            self.N_jobs = 1
            self.casm_project_dir = '.'
            print("WARNING: only one processor is allowed for running on your personal computer; CPUNum is overridden by 1")
                        
    ########################################

    def create_test_set_sobol(self,N_points,dim,bounds=[0.,1.],seed=1):

    # Create test set
        x_test = np.zeros((N_points,dim))
        eta = np.zeros((N_points,dim))
        i = 0
        while (i < N_points):
            x_test[i],seed = i4_sobol(dim,seed)
            x_test[i] = (bounds[1] - bounds[0])*x_test[i] + bounds[0] # shift/scale according to bounds
            eta[i] = np.dot(x_test[i],self.Q.T).astype(np.float32)
            if eta[i,0] <= 0.25:
                i += 1
        return x_test, eta, seed

    def create_test_set_billiardwalk(self,N_points,N_boundary=0):
        tau = 1
        eta, eta_b = billiardwalk(self.x0,self.n_planes,self.c_planes,N_points,tau)
        x_test = eta 
        self.x0 = eta[-1] # Take last point to be next initial point)
        eta = np.vstack((eta,eta_b[np.random.permutation(np.arange(len(eta_b)))[:N_boundary]]))

        return x_test,eta

    ########################################
    
    def ideal(self,x_test):

        T = self.T
        kB = 8.61733e-5
        invQ = np.linalg.inv(self.Q)
        mu_test = 0.25*kB*T*np.log(x_test/(1.-x_test)).dot(invQ)

        return mu_test

    def sample_wells(self, kappa,rnd):
        print('Sampling wells and end members...')
        etaW = self.Wells
        T = np.zeros((self.N_global_pts))
        Tavg = (self.Tmax - self.Tmin)/2 + self.Tmin
        for point in T:
            point = Tavg
        # define bias parameters
        if rnd==0:
            kappaW = etaW
        else:
            muW = self.idnn.predict([etaW,etaW,etaW,T,T,T])[1]
            muW[:,0] =  muW[:,0]/self.adjustedx
            for i in range(6):
                muW[:,i+1] = muW[:,i+1]/self.adjustedn
            kappaW = etaW + 0.5*muW/self.phi
        N_w = 25
        if self.test:
            N_w = 2
        kappaW = np.repeat(kappaW,N_w,axis=0)
        kappaW  += 0.15*(np.random.rand(*kappaW.shape)-0.5)

        # Sample between wells
        # Get vertices
        etaB = self.Vertices
        #print(etaB)
        if rnd==0:
            kappaB = etaB
        else:
            muB = self.idnn.predict([etaB,etaB,etaB,T,T,T])[1]
            muB[:,0] =  muB[:,0]/self.adjustedx
            for i in range(6):
                muB[:,i+1] = muB[:,i+1]/self.adjustedn
            kappaB = etaB + 0.5*muB/self.phi



        N_w2 = 20 # Number of random points per vertex
        if self.test:
            N_w2 = 2
        kappaW2 = np.zeros((2*(self.dim-1)*N_w2,self.dim))
        kappaW2[:,0] = kappaB[0,0]
        kappaW2 += 0.05*(np.random.rand(*kappaW2.shape)-0.5) # Small random perterbation
        for i in range(1,self.dim):
            for j in range(2*N_w2):
                kappaW2[2*(i-1)*N_w2 + j,i] = np.random.rand()*(kappaB[2*i-2,i] - kappaB[2*i-1,i]) + kappaB[2*i-1,i] # Random between positive and negative well

        kappa = np.vstack((kappa,kappaW,kappaW2))
        return kappa 

    ########################################

    def explore(self,rnd):
        
        # sample with sobol
        if self.Test_set == 'sobol':
            if rnd==0:
                x_bounds = [1.e-5,1-1.e-5]
            elif rnd<6:
                x_bounds = [-0.05,1.05]
            else:
                x_bounds = [0.,1.]
            x_test,eta,self.seed = self.create_test_set_sobol(self.N_global_pts,
                                                        self.dim,
                                                        bounds=x_bounds,
                                                        seed=self.seed)

        if self.Test_set == 'billiardwalk':
       # sample quasi-uniformly
            if rnd<6:
                N_b = int(self.N_global_pts/4)
            else:
                N_b = 0
            print('Create sample set...')
            x_test, eta = self.create_test_set_billiardwalk(self.N_global_pts,
                                    N_boundary=N_b)

         # define bias parameters
        if rnd==0:
            if self.Initial_mu == 'ideal':
                mu_test = self.ideal(x_test)
            else:
                mu_test = 0
        else:
            T = np.zeros((eta.shape[0],1))
            Tavg = (self.Tmax - self.Tmin)/2 + self.Tmin
            for point in T:
                point = Tavg
            mu_test = self.idnn.predict([eta,eta,eta,T,T,T])[1]
            mu_test[:,0] =  mu_test[:,0]*1/self.adjustedx
            for i in range(6):
                mu_test[:,i+1] = mu_test[:,i+1]/self.adjustedn
        
        kappa = eta + 0.5*mu_test/self.phi
        
        if 'Guided' in self.Sample_wells:
            kappa = self.sample_wells(kappa,rnd)   

        # submit casm
        print('Submit jobs to CASM...')
        submitCASM(self.N_jobs,self.phi,kappa,self.T,rnd,self.Account,self.Walltime,self.Mem,casm_project_dir=self.casm_project_dir,test=self.test,job_manager=self.job_manager,casm_version=self.CASM_version, data_generation=self.data_generation)
        print('Compile output...')
        compileCASMOutput(rnd, self.CASM_version, self.dim)            


   ########################################
        ##exploit hessian values
    def hessian(self,rnd, tol):
        kappa_test, eta, mu_load= loadCASMOutput(49,7,singleRnd=False)
        print('Predicting...')

        pred = self.idnn.predict(eta)
        free = pred[0]
        mu = pred[1]
        hessian= pred[2]


        eigen = np.zeros(eta.shape)
        eigenvector = np.zeros(hessian.shape)


        for i in range(len(hessian)):
            eigen[i,:], eigenvector[i,:,:] = LA.eig(hessian[i,:,:])

        def arg_zero_eig(e,tol = 0.1):
            return  np.sum(np.abs(e) < tol,axis=1) == e.shape[1]


        
        eigen = eigen/np.max(np.abs(eigen),axis=0)
        I = arg_zero_eig(eigen,tol)*(eta[:,0] > .45)*(eta[:,0] < .55)
        #print(I)
        #print(kappa_test)
        kappaE= kappa_test[I]
        #print(kappaE.size)
        print('tol',tol)
        print('kappaE', kappaE)
        #kappaE = random.shuffle(kappaE)
        #print('kappaE', kappaE)
        kappa_a = np.repeat(kappaE[:100],3,axis=0)
        kappa_b = np.repeat(kappaE[100:300],2,axis=0)
        kappa_local = np.vstack((kappa_a,kappa_b))
        kappa_local = 0.02*2.*(np.random.rand(*kappa_local.shape)-0.5) #perturb points randomly

        return kappa_local  
    ########################################
    
    def exploit(self,rnd):
        
        # local error
        print('Loading data...')
        kappa_test, eta_test, mu_test, T_test = loadCASMOutput(rnd-1,self.dim,singleRnd=True)
        T_test = (T_test - ((self.Tmax - self.Tmin)/2))/(self.Tmax - ((self.Tmax - self.Tmin)/2))
        print('Predicting...')
        mu_pred = self.idnn.predict([eta_test,eta_test,eta_test, T_test, T_test, T_test])[1]

        mu_pred[:,0] =  mu_pred[:,0]/self.adjustedx
        for i in range(6):
            mu_pred[:,i+1] = mu_pred[:,i+1]/self.adjustedn

        print('Finding high pointwise error...')
        error = np.sum((mu_pred - mu_test)**2,axis=1)
        kappaE =  kappa_test[np.argsort(error)[::-1]]
        
        
        # randomly perturbed samples
        if self.test:
            kappa_a = np.repeat(kappaE[:3],3,axis=0)
            kappa_b = np.repeat(kappaE[3:6],2,axis=0)
        else:
            kappa_a = np.repeat(kappaE[:200],3,axis=0)
            kappa_b = np.repeat(kappaE[200:400],2,axis=0)

        # sample wells 
        if 'exploit' in self.Sample_wells:
            etaW = find_wells(self.idnn,eta_test)
            muW = self.idnn.predict([etaW,etaW,etaW,T_test,T_test,T_test])[1]
            muW[:,0] =  muW[:,0]/self.adjustedx
            for i in range(6):
                muW[:,i+1] = muW[:,i+1]/self.adjustedn
            
            kappaW = etaW + 0.5*muW/(self.phi)
            if self.test:
                kappa_c = np.repeat(kappaW[:4],3,axis=0)
            else:
                kappa_c = np.repeat(kappaW[:400],4,axis=0)
            kappa_local = np.vstack((kappa_a,kappa_b, kappa_c))
        else:
            kappa_local = np.vstack((kappa_a,kappa_b))
        
        
        kappa_local += 0.02*2.*(np.random.rand(*kappa_local.shape)-0.5) #perturb points randomly

        ##add values from hessian
        tol = 0.035+0.001*i
        kappa_local += self.hessian(rnd, tol)
        
        # submit casm
        print('Submit jobs to CASM...')
        submitCASM(self.N_jobs,self.phi,kappa_local,self.T,rnd,self.Account,self.Walltime,self.Mem,casm_project_dir=self.casm_project_dir,test=self.test,job_manager=self.job_manager,casm_version=self.CASM_version, data_generation=self.data_generation)
        print('Compile output...')
        compileCASMOutput(rnd, self.CASM_version, self.dim)   

 

    ########################################
        
    # Define function creating IDNN with random hyperparameters
    def train_rand_idnn(self,rnd,set_i):
        learning_rate = np.power(10,(np.log10(self.LR_range[1]) - np.log10(self.LR_range[0]))*np.random.rand(1)[0] + np.log10(0.0001),dtype=np.float32)
        n_layers = np.random.randint(self.Layers_range[0],self.Layers_range[1]+1)
        hidden_units = n_layers*[np.random.randint(self.Neurons_range[0],self.Neurons_range[1]+1)]
        print(hidden_units)
        #activation = []
        #for i in range(n_layers):
        #    activation.append('relu')
        idnn = IDNN(self.dim,
                    hidden_units,
                    activation=self.activation,
                    transforms=self.IDNN_transforms(),
                    dropout=self.Dropout,
                    unique_inputs=True,
                    final_bias=True)
        self.opt = 'keras.optimizers.' + self.optimizer 
        idnn.compile(loss=['mse','mse',None],
                     loss_weights=[0.01,1,None],
                     optimizer=eval(self.opt)(lr=learning_rate))

        valid_loss,_ = self.surrogate_training()(rnd,idnn,f'_{set_i}')

        return hidden_units, learning_rate, valid_loss

    ########################################
        
    def hyperparameter_search(self,rnd):

        # submit
        commands = [f"sys.path.append('{os.path.dirname(__file__)}')",
                    'from active_learning import Active_learning']
        training_func = 'Active_learning("{}").train_rand_idnn'.format(self.config_path)
    
        self.hidden_units, self.lr = hyperparameterSearch(rnd,self.N_hp_sets,commands,training_func, self.job_manager,self.Account,self.Walltime,self.Mem,)

    ########################################
        
    def IDNN_transforms(self):

        sys.path.append(self.config_path)
        from TransformsModule import transforms 

        return transforms

    ########################################
    
    def surrogate_training(self):

        def training(rnd,idnn,set_i=''):
        # read in casm data
            print('Loading data...')
            _, eta_train, mu_train, T_train = loadCASMOutput(rnd,self.dim)
           
            # shuffle the training set (otherwise, the most recent results
            # will be put in the validation set by Keras)
            inds = np.arange(eta_train.shape[0])
            np.random.shuffle(inds)
            eta_train = eta_train[inds]
            mu_train = mu_train[inds]

            if self.WeightRecent == 'Yes':
                # weight the most recent high error points as high as all the other points
                n_points = len(eta_train)
                sample_weight = np.ones(n_points)
                if rnd > 0:
                    sample_weight[-1000:] = max(1,(n_points-1000)/(2*1000))
                sample_weight = sample_weight[inds]

            # create energy dataset (zero energy at origin)
            eta_train0 = np.zeros(eta_train.shape)
            g_train0 = np.zeros((eta_train.shape[0],1))
            T_train0 = np.zeros(T_train.shape)

            #Adjust mu, T from -1 to 1 
            T_train = (T_train - ((self.Tmax - self.Tmin)/2))/(self.Tmax - ((self.Tmax - self.Tmin)/2))
            #for i in range(self.Dim):
            #    eta_train[i] 

            # train
            lr_decay = self.LR_decay**rnd
            idnn.compile(loss=['mse','mse',None],
                            loss_weights=[0.01,1,None],
                            optimizer=eval(self.opt)(lr=self.lr*lr_decay))
            csv_logger = CSVLogger('training/training_{}{}.txt'.format(rnd,set_i),append=True)
            reduceOnPlateau = ReduceLROnPlateau(factor=0.5,patience=100,min_lr=self.Min_lr)
            callbackslist = [csv_logger, reduceOnPlateau]
            if EarlyStopping == 'Yes':
                earlyStopping = EarlyStopping(patience=150)
                callbackslist.append(earlyStopping)

            print('Training...')

            #mu_train[:,0] =  mu_train[:,0]*self.adjustedx
            #for i in range(6):
            #    mu_train[:,i+1] = mu_train[:,i+1]*self.adjustedn

            if self.WeightRecent == 'Yes':
                history =idnn.fit([eta_train0,eta_train,0*eta_train, T_train0, T_train, T_train0],
                      [g_train0,100*mu_train,0*mu_train],
                      validation_split=0.25,
                      epochs=self.Epochs,
                      batch_size=self.Batch_size,
                      sample_weight=[sample_weight,sample_weight],
                      callbacks=callbackslist)
            else:
                history=idnn.fit([eta_train0,eta_train,0*eta_train, T_train0, T_train, T_train0 ],
                      [g_train0,100*mu_train,0*mu_train],
                      validation_split=0.25,
                      epochs=self.Epochs,
                      batch_size=self.Batch_size)#,
                      #callbacks=callbackslist)

            print('Saving IDNN...')
            #idnn.save('idnn_{}{}'.format(rnd,set_i))

            valid_loss = history.history['val_loss'][-1]

            return valid_loss, idnn

        return training 

    ########################################
        
    def main_workflow(self):
        """
        Main function outlining the workflow.

        - Global sampling

        - Surrogate training (including hyperparameter search)

        - Local sampling
        """
        rnd = 250
        print('Train surrogate model, round ',rnd,'...')
        _, self.idnn = self.surrogate_training()(rnd,self.idnn)
        unique_inputs = self.idnn.unique_inputs

        print('Perform hyperparameter search...')
        self.hyperparameter_search(rnd)
        print('Load best model...')               
        unique_inputs = self.idnn.unique_inputs
        self.idnn = keras.models.load_model('idnn_1', custom_objects={'Transform': Transform(self.IDNN_transforms())})
        self.idnn.unique_inputs = unique_inputs
        
        import matplotlib.pyplot as plt

        kappa_Test, eta_train, mu_train, T = loadCASMOutput(250, 7)

        print('Predicting...')
        Tadjusted = (T - ((self.Tmax - self.Tmin)/2))/(self.Tmax - ((self.Tmax - self.Tmin)/2))
        mu_pred = 0.01*self.idnn.predict([eta_train,eta_train,eta_train, Tadjusted, Tadjusted, Tadjusted])[1]
        pred = plt.scatter(T, mu_pred[:,0], c = 'b')
        train = plt.scatter(T, mu_train[:,0], c = 'r')
#orig = plt.scatter(T, mu_orig[:,1], c = 'g')


        plt.show()




        """
        kappa_Test, eta_train, mu_train, T = loadCASMOutput(250, 7)
        eta_train0 = np.zeros(eta_train.shape)
        g_train0 = np.zeros((eta_train.shape[0],1))



        self.idnn.compile(loss=['mse','mse', None],loss_weights=[ .01, 10, None],optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001))
        Tadjusted = T -((T.max() - T.min())/2 + T.min())
        Tadjusted = Tadjusted/(Tadjusted.max())


        adjusted = np.zeros((2,7))
        adjusted[1,0] = 20000
        adjusted[1,1] = 500000

        for i in range(2):
            adjusted[0, i] = ((mu_train[:,i].max() - mu_train[:,i].min())/2 + mu_train[:,i].min())
            mu_train[:,i] = mu_train[:,i] - adjusted[0, i]
            #adjusted[1,i] = (mu_train[:,i].max())
            #mu_train[:,i] = 10*mu_train[:,i]/adjusted[1,i]
            mu_train[:,i] = adjusted[1,i]*mu_train[:,i]
        Tzero = np.zeros(T.shape)

        history=self.idnn.fit([ eta_train0,eta_train, 0*eta_train, Tzero,Tadjusted, Tzero],
                      [g_train0,mu_train,0*mu_train],
                      epochs=10000,
                      batch_size=20)

#model.save('6_20')
        import matplotlib.pyplot as plt

        print('Predicting...')
        mu_pred = self.idnn.predict([eta_train,eta_train,eta_train, Tadjusted, Tadjusted, Tadjusted])[1]
        pred = plt.scatter(T, mu_pred[:,0], c = 'b')
        train = plt.scatter(T, mu_train[:,0], c = 'r')
#orig = plt.scatter(T, mu_orig[:,1], c = 'g')


        plt.show()"""

        self.T = []
        temp = (self.Tmax-self.Tmin)/(self.N_jobs)
        for i in range(10):
            self.T.append(self.Tmin + i*temp)
        for rnd in range(self.N_rnds):
            print('Begin global sampling, round ',rnd,'...')
            self.explore(2*rnd)

            if rnd==1:
                print('Perform hyperparameter search...')
                self.hyperparameter_search(rnd)
                print('Load best model...')               
                unique_inputs = self.idnn.unique_inputs
                self.idnn = keras.models.load_model('idnn_1', custom_objects={'Transform': Transform(self.IDNN_transforms())})
                self.idnn.unique_inputs = unique_inputs
            print('Train surrogate model, round ',rnd,'...')
            _, self.idnn = self.surrogate_training()(rnd,self.idnn)
            unique_inputs = self.idnn.unique_inputs

            # Get rid of the memory leak
            keras.backend.clear_session()
            self.idnn = keras.models.load_model(f'idnn_{rnd}',
                                            custom_objects={'Transform': Transform(self.IDNN_transforms())})
            self.idnn.unique_inputs = unique_inputs
            print('Begin local sampling, round ',rnd,'...')
            print(2*rnd+1)
            self.exploit(2*rnd+1)

            ##hessian values

