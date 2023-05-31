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
import helperfunctions


class DataRecommender():

    def __init__(self,idnn,dictionary): 
        ## determine sdictionary ie 
        self.wells= #something
        self.N_global_pts
        self.idnn = idnn


    # def read():

    # def write():

    # def print():


    def find_wells(self,x,dim=4,bounds=[0,0.25],rereference=True):

    # Find "wells" (regions of convexity, with low gradient norm)

    # First, rereference the free energy
        if self.idnn.unique_inputs:
            pred = self.idnn.predict([x,x,x])
        else:
            pred = self.idnn.predict(x)
        mu_test = 0.01*pred[1]
        if rereference:
            eta_test = np.array([bounds[0]*np.ones(dim),
                                bounds[1]*np.ones(dim)])
            if self.idnn.unique_inputs:
                y = 0.01*self.idnn.predict([eta_test,eta_test,eta_test])[0]
            else:
                y = 0.01*self.idnn.predict(eta_test)[0]
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

        return eta[ind3]



    
    def sample_wells(self, rnd,N_w):
        # N_w Number of random points per vertex
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
  
        kappaW = np.repeat(kappaW,N_w,axis=0)
        kappaW  += 0.15*(np.random.rand(*kappaW.shape)-0.5)
        return kappaW 

        # Sample between wells
        
        
        
    def sample_vertices(self,rnd,N_w2):
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

        kappaW2 = np.zeros((2*(self.dim-1)*N_w2,self.dim))
        kappaW2[:,0] = kappaB[0,0]
        kappaW2 += 0.05*(np.random.rand(*kappaW2.shape)-0.5) # Small random perterbation
        for i in range(1,self.dim):
            for j in range(2*N_w2):
                kappaW2[2*(i-1)*N_w2 + j,i] = np.random.rand()*(kappaB[2*i-2,i] - kappaB[2*i-1,i]) + kappaB[2*i-1,i] # Random between positive and negative well
        return kappaW2
    
    def explore(self,rnd,test_set, x_bounds=[], temp=300):
        
        # sample with sobol
        if test_set == 'sobol':
            delta = (x_bounds[1] - x_bounds[0])*.05
            if rnd==0:
                x_bounds = [x_bounds[0]+ delta,x_bounds[1]-delta]
            elif rnd<6:
                x_bounds = [x_bounds[0]- delta,x_bounds[1]+delta]
            else:
                x_bounds = [x_bounds[0],bounds[1]]
            print('Create sample set...')
            x_test,eta,self.seed = self.create_test_set_sobol(self.N_global_pts,
                                                        self.dim,
                                                        bounds=x_bounds,
                                                        seed=self.seed)

        if test_set == 'billiardwalk':
       # sample quasi-uniformly
            if rnd<6:
                N_b = int(self.N_global_pts/4)
            else:
                N_b = 0
            print('Create sample set...')
            x_test, eta = self.create_test_set_billiardwalk(self.N_global_pts,
                                    N_boundary=N_b)



        T = np.ones(eta.shape[0])*temp
        if rnd==0:
            if self.Initial_mu == 'ideal':
                mu_test = self.ideal(x_test)
            else:
                mu_test = 0
        else:
            mu_test = self.idnn.predict([eta,eta,eta,T,T,T])[1]
        
        kappa = eta + 0.5*mu_test/self.phi

        return kappa          


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
    
    def high_error(self,rnd):
        
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
        