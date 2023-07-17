import sys, os

import numpy as np
import shutil
from shutil import copyfile

from importlib import import_module
import tensorflow as tf
from sobol_seq import i4_sobol
from active_learning.data_recommended.hitandrun import billiardwalk

from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from configparser import ConfigParser
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda
from numpy import linalg as LA
from active_learning.data_recommended.helperfunctions import convex, convexMult


class DataRecommender():

    def __init__(self,model,dictionary): 
        self.dict = dictionary
        self.model = model
        [self.domain, self.N_global_pts, self.wells,self.sample_well,self.sample_vertice] = self.dict.get_category_values('Sampling Domain')
        [self.input_alias,self.outputFolder] = self.dict.get_individual_keys(['input_alias','outputfolder'])
        [self.sample_hessian,self.hessian_repeat, self.hessian_repeat_points,self.sample_high_error, self.high_error_repeat, self.high_error_repeat_points] = self.dict.get_category_values('Exploit Parameters')
        self.header = ''

        [self.Model_type,    
         self.Data_Generation, self.Data_Generation_Source, self.restart,
         self.input_data,self.input_alias,self.output_alias,_,_, self.iterations,
         self.OutputFolder, self.seed, self.Input_dim, self.derivative_dim,
         self.output_dim,self.config_path] = self.dict.get_category_values('Overview')

        self.sampling_dict = self.dict.get_category('Sampling')



    def write(self,rnd,type,output):
        np.savetxt(self.outputFolder+'data/data_recommended/'+type+'_rnd'+str(rnd)+'.txt',output,fmt='%.12f',
                    header=self.header)
        
        if os.path.isfile(self.outputFolder+'data/data_recommended/'+type+'_rnd'+str(rnd)+'.txt'):
            allResults = np.loadtxt(self.outputFolder+'data/data_recommended/'+type+'_rnd'+str(rnd)+'.txt')
            output = np.vstack((allResults,output))
        np.savetxt(self.outputFolder+'data/data_recommended/'+type+'_rnd'+str(rnd)+'.txt',
                    output,
                    fmt='%.12f',
                    header=self.header)


        if os.path.isfile(self.outputFolder+'data/data_recommended/rnd'+str(rnd)+'.txt'):
            allResults = np.loadtxt(self.outputFolder+'data/data_recommended/rnd'+str(rnd)+'.txt')
            output = np.vstack((allResults,output))
        np.savetxt(self.outputFolder+'data/data_recommended/rnd'+str(rnd)+'.txt',
                    output,
                    fmt='%.12f',
                    header=self.header)

    # def print():

    def load_data(self,rnd,singleRnd=True):
        print('loading data')
        if singleRnd:
            input, input_non_derivative, output =  np.load(self.outputFolder + 'data/data_sampled/results{}.npy'.format(rnd),allow_pickle=True)
        else:
            input, input_non_derivative, output =  np.load(self.outputFolder + 'data/data_sampled/allResults{}.npy'.format(rnd),allow_pickle=True)
        
        j=0
        for i in range(np.size(self.input_alias)):
            _,_,derivative_dim,dimension,adjust = self.dict.get_category_values(self.input_alias[i])
            if derivative_dim:
                input[:,:,i] = (input[:,:,i]+adjust[0])*adjust[1]
            else:
                input_non_derivative[:,j] = (input_non_derivative[:,j]+adjust[0])*adjust[1]
                j+=1

        input = [input[0,:,:].T,input[1,:,:].T,input[2,:,:].T, input_non_derivative[0,:,:].T]

        for i in range(np.size(self.output_alias)):
            derivative,dimensions,adjust = self.dict.get_category_values(self.output_alias[i])
            output[:][derivative] = (output[:][derivative]+adjust[0])*adjust[1]
        
        return input,output

    def find_wells(self,x,T,dim=4,bounds=[0,0.25],rereference=True):

    # Find "wells" (regions of convexity, with low gradient norm)

    # First, rereference the free energy
        if self.model.unique_inputs:
            pred = self.model.predict([x,x,x,T])
        else:
            pred = self.model.predict(x,T)
        mu_test = 0.01*pred[1]
        if rereference:
            eta_test = np.array([bounds[0]*np.ones(dim),
                                bounds[1]*np.ones(dim)])
            if self.model.unique_inputs:
                y = 0.01*self.model.predict([eta_test,eta_test,eta_test,T])[0]
            else:
                y = 0.01*self.model.predict(eta_test)[0]
            g0 = y[0,0]
            g1 = y[1,0]
            mu_test[:,0] = mu_test[:,0] - 1./bounds[1]*(g1 - g0)
        gradNorm = np.sqrt(np.sum(mu_test**2,axis=-1))

        H = pred[2] # get the list of Hessian matrices
        ind2 = convexMult(H) # indices of points with local convexity
        eta = x[ind2]
        gradNorm = gradNorm[ind2]

        ind3 = np.argsort(gradNorm)
        
        # Return eta values with local convexity, sorted by gradient norm (low to high)

        return eta[ind3],T



    
    def sample_wells(self, rnd,N_w):
        # N_w Number of random points per vertex
        print('Sampling wells and end members...')
        etaW = self.Wells  
        etaW = np.repeat(etaW,N_w,axis=0)
        etaW  += 0.15*(np.random.rand(*etaW.shape)-0.5)
        return etaW 

        # Sample between wells
        
        
        
    def sample_vertices(self,rnd,N_w2):
        # Get vertices
        etaB = self.Vertices
        #print(etaB)

        etaW2 = np.zeros((2*(self.dim-1)*N_w2,self.dim))
        etaW2[:,0] = etaB[0,0]
        etaW2 += 0.05*(np.random.rand(*etaW2.shape)-0.5) # Small random perterbation
        for i in range(1,self.dim):
            for j in range(2*N_w2):
                etaW2[2*(i-1)*N_w2 + j,i] = np.random.rand()*(etaB[2*i-2,i] - etaB[2*i-1,i]) + etaB[2*i-1,i] # Random between positive and negative well
        return etaW2
    

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

    def create_test_set_billiardwalk(self,x0,n_planes,c_planes,N_points,N_boundary=0):
        tau = 1
        eta, eta_b = billiardwalk(x0,n_planes,c_planes,N_points,tau)
        x_test = eta 
        self.x0 = eta[-1] # Take last point to be next initial point)
        eta = np.vstack((eta,eta_b[np.random.permutation(np.arange(len(eta_b)))[:N_boundary]]))

        return x_test,eta
    

    def ideal(self,x_test):

        T = self.T
        kB = 8.61733e-5
        invQ = np.linalg.inv(self.Q)
        mu_test = 0.25*kB*T*np.log(x_test/(1.-x_test)).dot(invQ)

        return mu_test

    def explore(self,rnd,test_set, x_bounds=[]):

        output = np.ones((self.N_global_pts,1))
        outputorder = []

        for domain in self.sampling_dict['continuous_dependent'] :
            outputorder.append(self.sampling_dict['continuous_dependent'][domain]['values'])
            
        # sample with sobol
            if test_set == 'sobol':
                delta = (x_bounds[1] - x_bounds[0])*.05
                if rnd==0:
                    x_bounds = [x_bounds[0]+ delta,x_bounds[1]-delta]
                elif rnd<6:
                    x_bounds = [x_bounds[0]- delta,x_bounds[1]+delta]
                else:
                    x_bounds = [x_bounds[0],x_bounds[1]]
                print('Create sobol sample set...')
                x_test,eta,self.seed = self.create_test_set_sobol(self.N_global_pts,
                                                            self.sampling_dict['continuous_dependent'][domain]['dim'],
                                                            bounds=x_bounds,
                                                            seed=self.seed)

            # print(test_set)
            if test_set == 'billiardwalk':
        # sample quasi-uniformly
                if rnd<6:
                    N_b = int(self.N_global_pts/4)
                else:
                    N_b = 0
                print('Create billiard sample set')
                n_planes = self.sampling_dict['continuous_dependent'][domain]['n_planes']
                c_planes = self.sampling_dict['continuous_dependent'][domain]['c_planes']
                x_test, eta = self.create_test_set_billiardwalk(
                    self.sampling_dict['continuous_dependent'][domain]['x0'],
                    n_planes,c_planes,self.N_global_pts,
                                        N_boundary=N_b)
            output = np.hstack((output,eta[0:self.N_global_pts,:]))

        print('output1',np.shape(output))
        for domain in self.sampling_dict['continuous_independent']:
            outputorder.append([domain])
            range = self.sampling_dict['continuous_independent'][domain]
            # dim = self.sampling_dict['continuous_independent']
            random_continous = np.random.uniform(low=range[0], high=range[1], size=(self.N_global_pts,1))
            output = np.hstack((output,random_continous))


        for domain in self.sampling_dict['discrete']:
            outputorder.append([domain])
            range = self.sampling_dict['discrete'][domain]
            random_discrete = np.random.choice(range,self.N_global_pts)
            random_discrete = np.reshape(random_discrete,(self.N_global_pts,1))
            output = np.hstack((output,random_discrete))
        
        print('output2',np.shape(output))
        output = output[:,1:]
        self.write(rnd, test_set, output)       


    def relevent_columns(self,input):
        return np.hstack((input[1],input[3]))


   ########################################
        ##exploit hessian values
    def hessian(self,rnd,tol=0.1):
        input,output = self.load_data(rnd-1)
        print('Predicting...')

        # T_test_adjust = (T_test - ((self.Tmax - self.Tmin)/2))/(self.Tmax - ((self.Tmax - self.Tmin)/2))
        pred = self.model.predict(input)
        free = pred[0]
        mu = pred[1]
        hessian= pred[2]


        eigen = np.zeros(input[0].shape)
        eigenvector = np.zeros(hessian.shape)



        for i in range(len(hessian)):
            eigen[i,:], eigenvector[i,:,:] = LA.eig(hessian[i,:,:])

        def arg_zero_eig(e,tol = 0.1):
            return  np.sum(np.abs(e) < tol,axis=1) == e.shape[1]


        
        eigen = eigen/np.max(np.abs(eigen),axis=0)
        # print(kappa_test)
        I = arg_zero_eig(eigen,tol)#*(eta[:,0] > .45)*(eta[:,0] < .55)
        input_local = []


        for column in input:

            col = column[I]

            input_a = np.repeat(col[:self.hessian_repeat_points[0]],self.hessian_repeat[0],axis=0)
            input_b = np.repeat(col[self.hessian_repeat_points[0]:self.hessian_repeat_points[0]+self.hessian_repeat_points[1]],self.hessian_repeat[1],axis=0)
            input_local_col = np.vstack((input_a,input_b))
            input_local_col = 0.02*2.*(np.random.rand(*input_local_col.shape)-0.5) #perturb points randomly
            input_local.append(input_local_col)
        input_local = self.relevent_columns(input_local)

        # print(np.shape(input_local))
        print(np.shape(input_local)[0])

        if np.shape(input_local)[0] != 0:
            self.write(rnd, 'hessian', input_local)
        else:
            print('No data points in hessian tolerance')
        return input_local
    ########################################
    
    def high_error(self,rnd):
        
        # local error
        print('Loading data...')
        input,output = self.load_data(rnd-1)

        print('Predicting...')
        output_pred = self.model.predict(input)

        print('Finding high pointwise error...')

        # print(input)

        # input = np.array(input)

        input_local = []


        for i in range(np.size(self.output_alias)):
            derivative,dimensions,adjust = self.dict.get_category_values(self.output_alias[i])
            output[:][derivative] = (output[:][derivative]+adjust[0])*adjust[1]

            input_derivative = []


            error = np.sum((output_pred[derivative].T - output[:][derivative])**2,axis=1)
            
            for column in input:
                higherror =  column[np.argsort(error)[::-1],:]
            
            
            # randomly perturbed samples
                input_a = np.repeat(higherror[:self.high_error_repeat_points[0],:],self.high_error_repeat[0],axis=0)
                input_b = np.repeat(higherror[self.high_error_repeat_points[0]:self.high_error_repeat_points[0]+self.high_error_repeat_points[1],:],self.high_error_repeat[1],axis=0)
                

                input_local_col = np.vstack((input_a,input_b))
                input_local_col = (np.random.rand(*input_local_col.shape)-0.5) #perturb points randomly
                input_derivative.append(input_local_col)
            
            if input_local == []:
                input_local = input_derivative
            else:
                for i in range(len(input_local)):
                    input_local[i]  = np.vstack((input_local[i],input_derivative[i]))

        input_local = self.relevent_columns(input_local)
        self.write(rnd, 'high_error', input_local)

        return input_local
        