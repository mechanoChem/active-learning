import sys, os

import numpy as np
import shutil
from shutil import copyfile

from importlib import import_module
from data_generation_wrapper import submitCASM, compileCASMOutput, loadCASMOutput
import tensorflow as tf
from sobol_seq import i4_sobol
from hitandrun import billiardwalk

from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from configparser import ConfigParser
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda
from numpy import linalg as LA


class DataRecommender():

    def __init__(self,config_path): 
        ## determine dictionary ie 
        self.wells= #something

    def read():

    def write():

    def print():


    
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
    
        def explore(self,rnd,temp=300):
        
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
            print(self.N_global_pts)
            x_test, eta = self.create_test_set_billiardwalk(self.N_global_pts,
                                    N_boundary=N_b)
            # print(eta)
            # print(x_test)
         # define bias parameters
        T = np.ones(eta.shape[0])*temp
        print(T)
        if rnd==0:
            if self.Initial_mu == 'ideal':
                mu_test = self.ideal(x_test)
            else:
                mu_test = 0
        else:
            
            # T = np.zeros((eta.shape[0],1))
            # Tavg = (self.Tmax - self.Tmin)/2 + self.Tmin
            # for point in T:
            #     point = Tavg
            mu_test = self.idnn.predict([eta,eta,eta,T,T,T])[1]
            # mu_test[:,0] =  mu_test[:,0]*1/self.adjustedx
            # for i in range(6):
            #     mu_test[:,i+1] = mu_test[:,i+1]/self.adjustedn
        
        kappa = eta + 0.5*mu_test/self.phi
        
        if 'Guided' in self.Sample_wells:
            kappa = self.sample_wells(kappa,rnd)   

        # submit casm

        print('Submit jobs to CASM...')
        submitCASM(self.N_jobs,self.phi,kappa,T,rnd,self.Account,self.Walltime,self.Mem,casm_project_dir=self.casm_project_dir,test=self.test,job_manager=self.job_manager,casm_version=self.CASM_version, data_generation=self.data_generation)
        print('Compile output...')
        compileCASMOutput(rnd, self.CASM_version, self.dim,temp)            


   ########################################
        ##exploit hessian values
    def hessian(self,rnd, tol):
        kappa_test, eta, mu_load, T_test = loadCASMOutput(rnd,7,singleRnd=False)
        print('Predicting...')

        pred =  self.idnn.predict([eta,eta,eta, T_test, T_test, T_test])
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
        # print(kappa_test)
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

        ##Normalizing T to make it easier to train
        T_test_adjust = (T_test - ((self.Tmax - self.Tmin)/2))/(self.Tmax - ((self.Tmax - self.Tmin)/2))
        print('Predicting...')
        mu_pred = self.idnn.predict([eta_test,eta_test,eta_test, T_test_adjust, T_test_adjust, T_test_adjust])[1]

        mu_pred[:,0] =  mu_pred[:,0]/self.adjustedx
        for i in range(6):
            mu_pred[:,i+1] = mu_pred[:,i+1]/self.adjustedn

        print('Finding high pointwise error...')
        error = np.sum((mu_pred - mu_test)**2,axis=1)
        points = np.hstack((kappa_test, T_test))
        higherror =  points[np.argsort(error)[::-1],:]
        
        
        # randomly perturbed samples
        if self.test:
            kappa_a = np.repeat(points[:3,:],3,axis=0)
            kappa_b = np.repeat(points[3:6,:],2,axis=0)
        else:
            kappa_a = np.repeat(points[:200],3,axis=0)
            kappa_b = np.repeat(points[200:400],2,axis=0)

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
        
        Temp = kappa_local[:,self.dim]
        kappa_local = kappa_local[:,0:self.dim]
        kappa_local += 0.02*2.*(np.random.rand(*kappa_local.shape)-0.5) #perturb points randomly
        ##add values from hessian
        # tol = 0.035+0.001*i
        # hessian_values = self.hessian(rnd-1, tol)
        # print(np.shape(hessian_values))
        # print(np.shape(kappa_local))
        # kappa_local = np.vstack((kappa_local,hessian_values))
        
        # submit casm
        print('Submit jobs to CASM...')
        submitCASM(self.N_jobs,self.phi,kappa_local,Temp,rnd,self.Account,self.Walltime,self.Mem,casm_project_dir=self.casm_project_dir,test=self.test,job_manager=self.job_manager,casm_version=self.CASM_version, data_generation=self.data_generation)
        print('Compile output...')
        compileCASMOutput(rnd, self.CASM_version, self.dim)   