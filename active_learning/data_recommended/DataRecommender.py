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
        ## determine sdictionary ie 
        # self.wells= #something
        self.dict = dictionary
        self.model = model
        [self.domain, self.N_global_pts, self.wells,self.sample_well,self.sample_vertice,self.test_set,self.x0,self.Qpath,self.Initial_mu] = self.dict.get_category_values('Sampling Domain')
        [self.derivative_dim,T] = self.dict.get_individual_keys(['derivative_dim','temperatures'])
        self.Tmax = max(T)
        self.Tmin = min(T)
        [self.hessian_repeat, self.hessian_repeat_points, self.high_error_repeat, self.high_error_repeat_points] = self.dict.get_category_values('Exploit Parameters')
        self.Q = np.loadtxt(f'{self.Qpath}')
        self.invQ = np.linalg.inv(self.Q)[:,:self.derivative_dim]
        self.Q = self.Q[:self.derivative_dim]
        self.n_planes = np.vstack((self.invQ,-self.invQ))
        self.c_planes = np.hstack((np.ones(self.invQ.shape[0]),np.zeros(self.invQ.shape[0])))

    # def read():

    # def write():

    # def print():

    def load_single_rnd_output(self,rnd):
        kappa = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float32)[:,:self.derivative_dim]
        eta = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float32)[:,self.derivative_dim:2*self.derivative_dim]
        mu = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float32)[:,-self.derivative_dim:]
        T = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float32)[:,-self.derivative_dim-1:-self.derivative_dim]
        return eta,mu,T


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

    def create_test_set_billiardwalk(self,N_points,N_boundary=0):
        tau = 1
        eta, eta_b = billiardwalk(self.x0,self.n_planes,self.c_planes,N_points,tau)
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

    def explore(self,rnd,test_set, x_bounds=[], temp=300):
        
        # sample with sobol
        if test_set == 'sobol':
            delta = (x_bounds[1] - x_bounds[0])*.05
            if rnd==0:
                x_bounds = [x_bounds[0]+ delta,x_bounds[1]-delta]
            elif rnd<6:
                x_bounds = [x_bounds[0]- delta,x_bounds[1]+delta]
            else:
                x_bounds = [x_bounds[0],x_bounds[1]]
            print('Create sample set...')
            x_test,eta,self.seed = self.create_test_set_sobol(self.N_global_pts,
                                                        self.dim,
                                                        bounds=x_bounds,
                                                        seed=self.seed)

        # print(test_set)
        if test_set == 'billiardwalk':
       # sample quasi-uniformly
            if rnd<6:
                N_b = int(self.N_global_pts/4)
            else:
                N_b = 0
            print('Create sample set...')
            x_test, eta = self.create_test_set_billiardwalk(self.N_global_pts,
                                    N_boundary=N_b)
            # print('eta ',eta)


        # T = np.ones(eta.shape[0])*temp
        # if rnd==0:
        #     if self.Initial_mu == 'ideal':
        #         mu_test = self.ideal(x_test)
        #     else:
        #         mu_test = 0
        # else:
        #     mu_test = self.idnn.predict([eta,eta,eta,T,T,T])[1]
        
        # kappa = eta + 0.5*mu_test/self.phi

        return eta         




   ########################################
        ##exploit hessian values
    def hessian(self,rnd, tol,repeat):
        eta, mu_load, T_test = self.load_single_rnd_output(rnd)
        print('Predicting...')

        T_test_adjust = (T_test - ((self.Tmax - self.Tmin)/2))/(self.Tmax - ((self.Tmax - self.Tmin)/2))
        pred = self.idnn.predict([eta,eta,eta, T_test_adjust, T_test_adjust, T_test_adjust])[1]
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
        eta = eta[I]
        # kappaE= kappa_test[I]
        # #print(kappaE.size)
        # print('tol',tol)
        # print('kappaE', kappaE)
        #kappaE = random.shuffle(kappaE)
        #print('kappaE', kappaE)
        eta_a = np.repeat(eta[:self.hessian_repeat_points[0]],self.hessian_repeat[0],axis=0)
        eta_b = np.repeat(eta[self.hessian_repeat_points[0]:self.hessian_repeat_points[0]+self.hessian_repeat_points[1]],self.hessian_repeat[1],axis=0)
        eta_local = np.vstack((eta_a,eta_b))
        eta_local = 0.02*2.*(np.random.rand(*eta_local.shape)-0.5) #perturb points randomly

        return eta_local
    ########################################
    
    def high_error(self,rnd):
        
        # local error
        print('Loading data...')
        eta_test, mu_test, T_test = self.load_single_rnd_output(rnd)

        ##Normalizing T to make it easier to train
        T_test_adjust = (T_test - ((self.Tmax - self.Tmin)/2))/(self.Tmax - ((self.Tmax - self.Tmin)/2))
        print('Predicting...')
        mu_pred = self.idnn.predict([eta_test,eta_test,eta_test, T_test_adjust])[1]

        mu_pred[:,0] =  mu_pred[:,0]/self.adjustedx
        for i in range(6):
            mu_pred[:,i+1] = mu_pred[:,i+1]/self.adjustedn

        print('Finding high pointwise error...')
        error = np.sum((mu_pred - mu_test)**2,axis=1)
        points = np.hstack((eta_test, T_test))
        higherror =  points[np.argsort(error)[::-1],:]
        
        
        # randomly perturbed samples
        eta_a = np.repeat(points[:self.high_error_repeat_points[0],:],self.high_error_repeat[0],axis=0)
        eta_b = np.repeat(points[self.high_error_repeat_points[0]:self.high_error_repeat_points[0]+self.high_error_repeat_points[1],:],self.high_error_repeat[1],axis=0)
         

        eta_local = np.vstack((eta_a,eta_b))
        eta_local = 0.02*2.*(np.random.rand(*eta_local.shape)-0.5) #perturb points randomly

        return eta_local
        