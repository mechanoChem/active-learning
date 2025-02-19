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
import pandas as pd
from scipy.spatial.distance import cdist
import json
from math import isnan
from sklearn.metrics import mean_squared_error

class DataRecommender():

    def __init__(self,model,dictionary): 
        self.dict = dictionary
        self.model = model
        [self.N_global_pts, self.sample_external,self.external_path,
         self.external_points,self.external_perturbation] = self.dict.get_category_values('Explore_Parameters')

        [self.sample_non_convexities,self.non_convexities_repeat, self.non_convexities_repeat_points,self.non_convexities_perturb,self.sample_high_error,
        self.high_error_repeat, self.high_error_repeat_points, self.high_error_perturb, self.sample_find_wells, self.wells_repeat,
        self.wells_repeat_points,self.wells_perturb, self.lowest_free_energy,self.lowest_repeat,self.lowest_repeat_points, self.lowest_free_energy_file,self.lowest_perturb,
        self.sample_sensitivity,self.sensitivity_repeat, self.sensitivity_repeat_points,self.sensitivity_perturb,self.QBC,self.QBC_repeat,self.QBC_repeat_points,self.QBC_perturb] = self.dict.get_category_values('Exploit_Parameters')
        self.header = ''

        [self.Model_type,    
         self.Data_Generation, self.Data_Generation_Source, self.restart,
         self.inputs_data,self.inputs_alias,self.output_alias, self.iterations,
         self.OutputFolder, self.inputs_dim, self.derivative_dim,
         self.output_dim,self.config_path,self.T,self.testing_set,self.graph,self.reweight,self.alpha,self.prediction_points, self.novelty] = self.dict.get_category_values('Main')



        self.sampling_dict = self.dict.get_category('Sampling')
        self.Qinv = self.sampling_dict['continuous_dependent']['etasampling']['invQ']
        self.min_param=.001
        self.type_criterion= {}
        if os.path.isfile(self.OutputFolder+'data/data_recommended/types.txt'):
            with open(self.OutputFolder+'data/data_recommended/types.txt') as f: 
                data = f.read() 
            self.type_criterion = json.loads(data)
        self.stopping_alpha=0
        self.total_global=0
        self.dim=self.inputs_dim-1  
        for domain in self.sampling_dict['continuous_dependent'] :
            self.x0 =self.sampling_dict['continuous_dependent'][domain]['x0']   
            self.domain2d = self.sampling_dict['continuous_dependent'][domain]['2d'] 
            self.x0_2d = self.x0[0:2]
            self.points_2d = self.sampling_dict['continuous_dependent'][domain]['2d_points'] 

        [self.relevent_indices]= self.dict.get_individual_keys(self.Data_Generation_Source,['relevent_indices'])

    def explore2d(self):

        return self.domain2d
    
    def points2D(self):
        return self.points_2d
    
    def create_types(self,type):   
        self.type_criterion[type] = len(self.type_criterion)

        self.weights = np.ones(len(self.type_criterion))
        with open(self.OutputFolder+'data/data_recommended/types.txt', 'w') as file: 
            file.write(json.dumps(self.type_criterion))

    def write(self,rnd,type,output):
        value=self.type_criterion[type]
        output = self.keep_good_output(output)
        output = np.hstack((output,self.type_criterion[type]*np.ones((np.shape(output)[0],1))))


        np.savetxt(self.OutputFolder+'data/data_recommended/'+type+'_rnd'+str(rnd)+'.txt',output,fmt='%.12f',
                    header=self.header)
                    

        if os.path.isfile(self.OutputFolder+'data/data_recommended/initial_rnd'+str(rnd)+'.txt'):
            allResults = np.loadtxt(self.OutputFolder+'data/data_recommended/initial_rnd'+str(rnd)+'.txt')
            if allResults.size != 0:  # allResults is not empty
                output = np.vstack((allResults, output))
        np.savetxt(self.OutputFolder+'data/data_recommended/initial_rnd'+str(rnd)+'.txt',
                    output,
                    fmt='%.12f',
                    header=self.header)
    
    def subtract_largest(self,arr, value=0.15):
            # Iterate through each row of the array
            for row in arr:
                # Find the index of the largest element in the row
                max_index = np.argmax(np.abs(row[1:7]))
                if row[max_index+1] > 0.2:
                    row[max_index+1] -= value*np.random.random()
                if row[max_index+1] < -0.2:
                    row[max_index+1] += value*np.random.random()
                if np.abs(row[max_index+1]) <= 0.2:
                    row[max_index+1] = row[max_index+1]/4
                # print('new row',row)
            return arr

    def sample_wells(self, rnd):
        # self.sample_external_data(rnd,self.wells,[.15,0.5],'sample_wells')
        etaW = np.zeros((2*self.dim+1,self.dim))
        #added an extra point to be at (0.5,0)

        # wells
        etaW[:,0] = 0.5
        # etaW[2,1] =0.425
        for i in range(1,self.dim):
            etaW[2*i,i] = 0.475
            etaW[2*i+1,i] = -0.475
        # end members
        etaW[0,0] = .075
        etaW[1,0] = .925
        # etaW[:,0] = np.linspace(0.1,0.9,14)

        # define bias parameters
        if rnd<100:
            kappaW = etaW
        else:
            phi = np.array([10.0,0.1,0.1,0.1,0.1,0.1,0.1])
            muW = self.model.predict([etaW,T])[1]
            kappaW = etaW + 0.5*muW/phi
        N_w = round(self.external_points*self.weights[self.type_criterion['sample_wells']])  #35 #50
        if N_w < 1:
            N_w=1
        kappaW = np.repeat(kappaW,N_w,axis=0)
        kappaW[:,0]  = kappaW[:,0]+ .05*(np.random.rand(*kappaW[:,0].shape)-0.5)
        for i in range(1,self.dim):
            #only perturb points that are already set to 0.425
            kappaW[:,i]  = kappaW[:,i]+ .1*np.abs(kappaW[:,i])/.425*(np.random.rand(*kappaW[:,i].shape)-0.5)

        kappaW[N_w*2:,1:] += .005*rnd*(np.random.rand(*kappaW[N_w*2:,1:].shape)-0.5)

        # #perturb points near x=0 by plus or minus x
        kappaW[0:N_w,1:]  += 2*kappaW[0:N_w,0:1]*(np.random.rand(*kappaW[0:N_w,1:].shape)-0.5)
        # perturb points near x=1 by plus or minus (1-x)
        kappaW[N_w:N_w*2,1:] += 2*(1-kappaW[N_w:N_w*2,0:1])*(np.random.rand(*kappaW[N_w:N_w*2,1:].shape)-0.5)



        kappaW  = kappaW+ self.external_perturbation*(np.random.rand(*kappaW.shape)-0.5)

        temp = self.T*np.ones(((2*self.dim+1)*N_w,1))

        

        I =  [not item for item in self.find_good_output(kappaW)] #points that are not in bounds
        i=0
        while i < 5 and sum(I) >0:
            i+=1
            kappaW[I,:] = self.subtract_largest(kappaW[I,:])
            I =  [not item for item in self.find_good_output(kappaW)]

        input_local = np.hstack((kappaW,temp))
        random_indices = np.random.choice(input_local.shape[0], size=N_w, replace=False)
        random_subset = input_local[random_indices]
        
        
        self.write(rnd, 'sample_wells', random_subset)
           
    def sample_vertices(self,rnd):

        N_w2 = round(self.external_points*self.weights[self.type_criterion['sample_vertices']]) # Number of random points per vertex
        if N_w2<1:
            N_w2=1
        kappaW2 = np.zeros(((self.dim-1)*N_w2,self.dim))
        temp = self.T*np.ones(((self.dim-1)*N_w2,1))
        kappaW2[:,0] = 0.5
        kappaW2 +=.005*rnd*(np.random.rand(*kappaW2.shape)-0.5) #random perturbation
     
        # between += 0.5

        for i in range(1,self.dim):

            kappaW2[(i-1)*N_w2:i*N_w2,i] = (np.random.rand(*kappaW2[(i-1)*N_w2:i*N_w2,i].shape)-0.5)  # Random between positive and negative well (ie pos and neg 0.5)
        
        I =  [not item for item in self.find_good_output(kappaW2)] #points that are not in bounds
        # print('I',I)
        i=0
        j=sum(I)
        while i < 5 and sum(I) >0:
            # print('sample vertices finding more inbound points')
            i+=1
            kappaW2[I,:] = kappaW2[I,:]+ .05*(np.random.rand(*kappaW2[I,:].shape)-0.5)
            I =  [not item for item in self.find_good_output(kappaW2)]
            # print('I',I)
        
        
        input_local = np.hstack((kappaW2,temp))
        random_indices = np.random.choice(input_local.shape[0], size=N_w2, replace=False)
        random_subset = input_local[random_indices]


        self.write(rnd, 'sample_vertices', random_subset)


    def get_types(self):
        return self.type_criterion

    def load_data(self,rnd,singleRnd=True):
        print('Loading Data for rnd', rnd)
        if singleRnd:
            data =  np.load(self.OutputFolder + 'data/data_sampled/results{}.npy'.format(rnd),allow_pickle=True)
        else:
            data =  np.load(self.OutputFolder + 'data/data_sampled/allResults{}.npy'.format(rnd),allow_pickle=True)
        
        # print('data line 74',data)
        
        inputs, output = self.model.array_to_column(data)

        return inputs,output
    

    #given a set of points, should perturb them if applicable
    #then write them
    #input is a list, where each entry is an array corresponding to 
    #different input variables. The array is ordered by importance
    #repeat is an array with the number of search points
    #duplicate is an array of same size of repeat, saying how many times
    #each point should be duplicated

    def find_new_points(self,inputs,repeat,duplicate,perturb,type_criterion):
        input_local = []
        for i in range(np.size(repeat)):
            repeat[i] = round(repeat[i]*self.weights[self.type_criterion[type_criterion]])
            if repeat[i] < 10:
                repeat[i] = 10 
        for k in range(np.size(self.type_criterion_of_input)):
            col = inputs[k]
            if (np.shape(col)[0]==np.size(col)):
                col = np.reshape(col,(np.size(col),1) )
            
            start = 0
            for i in range(np.size(repeat)):
                if i==0:
                    input_local_col = np.repeat(col[start:repeat[i]][:],duplicate[i],axis=0)
                    start=repeat[i]
                else:
                    new_values = np.repeat(col[start:start+repeat[i],:],duplicate[i],axis=0)
                    start = start+repeat[i]
                    input_local_col = np.vstack((input_local_col,new_values))

            if self.type_criterion_of_input[k] == 0 or self.type_criterion_of_input[k] == 1: 
                #EDIT to include perturbation option for discrete
                input_local_col += perturb[0]*(np.random.rand(*input_local_col.shape)-perturb[1]) #perturb points randomly
            
            input_local.append(input_local_col)
        return input_local


    def combine_list(self, points):
        array = np.array([])
        for i in range(len(points)):
            col = points[i]
            if i== 0:
                array = points[i]
            else:
                array =np.hstack((array,points[i]))
        return array


    def find_good_output(self,eta):
        Qinv = self.sampling_dict['continuous_dependent']['etasampling']['invQ'] 

        keep = np.ones((np.shape(eta)[0]), dtype=np.int8)


        for k in range(np.shape(eta)[0]):
            isbad = False 
            badvalue = 0
            for i in range(32):
                value = 0
                for j in range(self.dim):
                    value += Qinv[i,j]*eta[k,j]
                if value > 1.05 or value < -0.05:
                    badvalue=value
                    isbad = True
            if isbad == True:
                keep[k]=0
        true_false_array = [bool(x) for x in keep]
        return true_false_array


    def keep_good_output(self,output):
        eta = output[:,0:self.dim]
        true_false_array = self.find_good_output(eta)
        return  output[true_false_array,:]

    def find_wells(self,rnd,dim=4,bounds=[0,0.25],rereference=True):

        print('Finding wells rnd', rnd)

    # Find "wells" (regions of convexity, with low gradient norm)
        

        inputs = self.inputs_explore.copy()
        outputs= self.output_explore.copy()
        eigenvalues = self.eigenvalues.copy()
        mu = outputs[1]

        gradNorm = np.sqrt(np.sum(mu**2,axis=1))

        H = outputs[2] # get the list of Hessian matrices
        ind2 = convexMult(H) # indices of points with local convexity
        # eta = x[ind2]

        ## only keep points which have local convexity
        gradNorm = gradNorm[ind2]
        inputs = inputs[ind2]

        #sort points by the gradNorm
        I = np.argsort(gradNorm)

        inputs = inputs[I,:]
        input =[inputs[:,0:self.dim],inputs[:,self.dim:]]

        
        # for k in range(np.size(self.type_criterion_of_input)):
        #     column = inputs[k]
        #     inputs[k] = column[I]
        
        input_local = self.find_new_points(input,self.wells_repeat_points,self.wells_repeat,[self.wells_perturb,.5],'find_wells')
        input_local = self.combine_list(input_local)



        if np.shape(input_local)[0] != 0:
            self.write(rnd, 'find_wells', input_local)
        else:
            print('No data points near wells')


        return input_local
    

    def sample_external_data(self,rnd):

        eta = np.loadtxt(self.external_path)
        T= self.T*np.ones((np.shape(eta)[0],1))
        input =[eta,T]
        input_local = self.find_new_points(input,[np.shape(eta)[0]],[self.external_points],[self.external_perturbation,.5],'Externally_supplied_points')
        input_local = self.combine_list(input_local)



        self.write(rnd, 'Externally_supplied_points', input_local)


    def create_test_set_sobol(self,N_points,dim,bounds=[0.,1.],seed=1):

    # Create test set
        x_test = np.zeros((N_points,dim))
        eta = np.zeros((N_points,dim))
        i = 0
        while (i < N_points):
            x_test[i],seed = i4_sobol(dim,seed)
            x_test[i] = (bounds[1] - bounds[0])*x_test[i] + bounds[0] # shift/scale according to bounds
            eta[i] = np.dot(x_test[i],self.Q.T).astype(np.float64)
            if eta[i,0] <= 0.25:
                i += 1
        return x_test, eta, seed

    def create_test_set_billiardwalk(self,x0,n_planes,c_planes,N_points,N_boundary=0):
        tau = 1
        eta, eta_b = billiardwalk(x0,n_planes,c_planes,N_points,tau)
        x_test = eta 
        if len(eta_b)>0:
            eta = np.vstack((eta[np.random.permutation(np.arange(len(eta)))[:N_points-N_boundary]],eta_b[np.random.permutation(np.arange(len(eta_b)))[:N_boundary]]))
        else:
            eta= eta[np.random.permutation(np.arange(len(eta)))]
        #return next point, to the first point in next round, and eta
        return eta[-1],eta
    



    def create_boundary_points(self,x0,n_planes,c_planes,N_points,N_boundary=0):
        tau = 1
        eta, eta_b = billiardwalk(x0,n_planes,c_planes,N_points,tau)
        return eta_b

    def ideal(self,x_test):

        T = self.T
        kB = 8.61733e-5
        invQ = np.linalg.inv(self.Q)
        mu_test = 0.25*kB*T*np.log(x_test/(1.-x_test)).dot(invQ)

        return mu_test
    
    def construct_input_types(self):
        [self.type_criterion_of_input] = self.dict.get_individual_keys('Ordering',['type_of_input'])

    def explore_existing(self,rnd):
        global_points = round(self.N_global_pts*self.weights[self.type_criterion['billiardwalk']]) 
        
        database = np.opentxt('billiardwalk_points.txt')
        data = database[self.global_points:self.global_points+global_points,:]
        output = np.hstack((data[:,0:7],data[:,-8:-7])) #eta's and temp

        self.global_points += global_points
        self.write(rnd, "billiardwalk", output)   
         

    def explore(self,rnd,twoD=False):


        if twoD:
            global_points = round(self.N_global_pts*self.points_2d*self.weights[self.type_criterion['billiardwalk_2d']]) 
        else:
            global_points = round(self.N_global_pts*self.weights[self.type_criterion['billiardwalk']]) 
        if global_points < 10:
            global_points=10
        output = np.ones((global_points,1))
        outputorder = []
       
        for domain in self.sampling_dict['continuous_dependent'] :
            test_set = self.sampling_dict['continuous_dependent'][domain]['type']

            
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
                x_test,eta,self.seed = self.create_test_set_sobol(global_points,
                                                            self.sampling_dict['continuous_dependent'][domain]['dim'],
                                                            bounds=x_bounds,
                                                            seed=self.seed)

            # print(test_set)
            if test_set == 'billiardwalk':
        # sample quasi-uniformly
                if rnd<6:
                    N_b = round(global_points/4)
                else:
                    N_b = 0
                print('Create billiard sample set')
                if twoD:
                    n_planes = self.sampling_dict['continuous_dependent'][domain]['n_planes_2d']
                    c_planes = self.sampling_dict['continuous_dependent'][domain]['c_planes_2d']
                    perturb = self.sampling_dict['continuous_dependent'][domain]['2d_perturb']
                    self.x0_2d, eta_2d = self.create_test_set_billiardwalk(self.x0_2d,n_planes,c_planes,global_points,N_boundary=N_b)
                    otheretas= np.zeros((np.shape(eta_2d)[0],self.dim-2))
                    # if rnd>5:
                    otheretas += perturb*(np.random.rand(*otheretas.shape)-.5)
                    eta = np.hstack((eta_2d,otheretas))
                else:
                    n_planes = self.sampling_dict['continuous_dependent'][domain]['n_planes']
                    c_planes = self.sampling_dict['continuous_dependent'][domain]['c_planes']
                    self.x0, eta = self.create_test_set_billiardwalk(self.x0,n_planes,c_planes,global_points,N_boundary=N_b)
            output = np.hstack((output,eta[0:global_points,:]))
        for domain in self.sampling_dict['continuous_independent']:
            range = self.sampling_dict['continuous_independent'][domain]
            [dim] = self.dict.get_individual_keys(domain,['dimensions'])
            random_continous = np.random.uniform(low=range[0], high=range[1], size=(global_points,dim))
            output = np.hstack((output,random_continous))
        

        for domain in self.sampling_dict['discrete']:
            range = self.sampling_dict['discrete'][domain]
            [dim] = self.dict.get_individual_keys(domain,['dimensions'])
            random_discrete = np.random.choice(range,global_points*dim)
            random_discrete = np.reshape(random_discrete,(global_points,dim))
            output = np.hstack((output,random_discrete))
        

        output = output[:,1:]
         ## EDIT - temporary 



        if twoD:
            self.write(rnd, "billiardwalk_2d", output)  
        else:
            self.write(rnd, test_set, output)       


    def relevent_columns(self,inputs):
        return np.hstack((inputs[0],inputs[1]))


    def find_eigenvalues_explore(self,rnd):
        inputs = self.inputs_explore.copy()
        pred = self.output_explore.copy()
        free = pred[0]
        mu = pred[1]
        hessian= pred[2]
        eigen = np.zeros(mu.shape)
        eigenvector = np.zeros(hessian.shape)



        for i in range(len(hessian)):
            eigen[i,:], eigenvector[i,:,:] = LA.eig(hessian[i,:,:])
        
        self.eigenvalues = eigen

   

    def get_latest_pred(self,rnd):
        self.inputs,self.output = self.load_data(rnd-1)
        print('Predicting for rnd',rnd)
        self.output_pred = self.model.predict(self.inputs.copy())
    


    def high_error(self,rnd):
        
        

        print('Finding high pointwise error...')

        inputs = self.inputs.copy()
        output = self.output.copy()
        output_pred = self.output_pred.copy()

        input_local = []

        for i in range(np.size(self.output_alias)):
            derivative,dimensions,adjust = self.dict.get_category_values(self.output_alias[i])
            # output[:][derivative] = (output[:][derivative]+adjust[0])*adjust[1]

            input_derivative = []


            error = np.sum((output_pred[derivative] - output[:][i])**2,axis=1)
            for k in range(np.size(self.type_criterion_of_input)):
                column = inputs[k]
                higherror =  column[np.argsort(error)[::-1],:]
                inputs[k]=higherror
            
            input_local = self.find_new_points(inputs, self.high_error_repeat_points, self.high_error_repeat,[self.high_error_perturb,0.5], 'high_error')
   
        input_local = self.combine_list(input_local)

        self.write(rnd, 'high_error', input_local)

        return input_local
    
    def isindomain(self,eta):

        output = []
        domain = True
        for eta_row in eta: 
            domain=True
            for row in self.Qinv:
                total = np.sum(row*eta_row)
                if (total<0):
                    domain=False
                    break
                if (total>1):
                    domain=False
                    break
            output.append(domain)
        return output

    def lowest_free_energy_curve(self,rnd):
        print('Finding lowest free energy curve rnd',rnd)

        eta = np.loadtxt(self.lowest_free_energy_file)
        T= self.T*np.ones((np.shape(eta)[0],1))
        pred_new = self.model.predict([eta,T])
        free_new = pred_new[0]
        mu_new = pred_new[1]

        data = np.hstack((eta,mu_new,free_new))

        data_list ={'x': data[:,0],'free': free_new[:,0]}
        resultlist = ['x']
        
        for i in range(1,self.dim):
            resultlist.append('eta_{}'.format(i))
            data_list['eta_{}'.format(i)] = data[:,i]


        df = pd.DataFrame(data_list)

        # Group by x1 and find the indices of the minimum y values within each group
        idx = df.groupby('x')['free'].idxmin()


        result = df.loc[idx, resultlist]
        result= result.sample(frac=1).reset_index(drop=True)
        temp = self.T*np.ones((np.shape(result)[0],1))
        input =[result,temp]
        input_local = self.find_new_points(input,self.lowest_repeat_points,self.lowest_repeat,[self.lowest_perturb,.5],'lowest_free_energy')
        input_local = self.combine_list(input_local)

        self.write(rnd, 'lowest_free_energy', input_local)




    # Function to sort n2, n3, and n4 in each point from highest to lowest
    def sort_order_parameters(self,points):
        sorted_points = []
        for point in points:
            n1 = point[0]
            n_rest_sorted = sorted(point[1:], reverse=True)
            sorted_points.append([n1] + n_rest_sorted)
        return np.array(sorted_points)

    # Function to calculate novelty scores
    def calculate_novelty_score_distance(self,new_points, old_points, k=5):

        #calculate distances
        distances = cdist(new_points, old_points)
        # Use the average distance to the k-nearest neighbors as the novelty score
        novelty_scores_dist = np.mean(np.sort(distances, axis=1)[:, :k], axis=1)
        shortest_distances = np.min(distances, axis=1)
        novelty_scores_dist[shortest_distances < self.min_param] = 0

        pred_new = self.model.predict([new_points[:,:self.dim],new_points[:,self.dim:]])
        free_new = pred_new[0]
        mu_new = pred_new[1]
        hessian_new = pred_new[2]

        eigen = np.zeros(mu_new.shape)
        eigenvector = np.zeros(hessian_new.shape)


        for i in range(len(hessian_new)):
            eigen[i,:], eigenvector[i,:,:] = LA.eig(hessian_new[i,:,:])

        eigen_max = np.max(np.abs(eigen), axis=1)
        novelty_scores=eigen_max*novelty_scores_dist



        return novelty_scores
    
    def read_initial_recommended(self,rnd):
        return np.loadtxt(self.OutputFolder+'data/data_recommended/initial_rnd'+str(rnd)+'.txt')
    
    def read_recommended(self,rnd):
        return np.loadtxt(self.OutputFolder+'data/data_recommended/rnd'+str(rnd)+'.txt')

    def append(self,rnd,newpoints,onlyInitial=False):
        # initial = np.loadtxt(self.OutputFolder+'data/data_recommended/initial_rnd'+str(rnd)+'.txt')
        # output = np.vstack((initial,newpoints))
        # Append newpoints to the initial_rnd file
        if newpoints is not None:
            with open(self.OutputFolder + 'data/data_recommended/initial_rnd' + str(rnd) + '.txt', 'a') as f:
                np.savetxt(f, newpoints, fmt='%.12f')

            if not onlyInitial:
            # Append newpoints to the rnd file
                with open(self.OutputFolder + 'data/data_recommended/rnd' + str(rnd) + '.txt', 'a') as f:
                    np.savetxt(f, newpoints, fmt='%.12f')

    def choose_points(self,rnd):
        kept_points=[]
        if not os.path.exists(self.OutputFolder+'data/data_recommended/initial_rnd'+str(rnd)+'.txt'):
            output = np.empty((0,15))
            np.savetxt(self.OutputFolder+'data/data_recommended/initial_rnd'+str(rnd)+'.txt',
                    output,
                    fmt='%.12f',
                    header=self.header)
        elif self.novelty ==True and rnd>0:
            print('Begin Reduction of Points, round ',rnd,'...')
            new_points = self.read_initial_recommended(rnd)
            old_points = self.read_recommended(0)
            for i in range(1,rnd):
                old_points=np.vstack((old_points,self.read_recommended(i)))
            novelty_scores = self.calculate_novelty_score_distance(new_points[:,:-1],old_points[:,:-1])
            normalized_scores = novelty_scores / np.sum(novelty_scores)
            n = len(novelty_scores)

            import matplotlib.pyplot as plt
            indices = np.arange(n)
            np.random.shuffle(indices)  # Randomly shuffle the indices
            
            kept_points.append(new_points[indices[0],:])
            j=0
            kept_scores=[]
            kept_indices=[]
            kept_scores.append(normalized_scores[indices[0]])
            kept_indices.append(0)

            for i in range(1,n):
                prev_score = normalized_scores[indices[j]]
                curr_score = normalized_scores[indices[i]]
                diff = curr_score-prev_score
                # compare = np.random.rand()*np.average(normalized_scores)
                compare = np.random.rand()*np.max(normalized_scores)
                if (curr_score > prev_score or curr_score > compare) and curr_score != 0:
                    kept_points.append(new_points[indices[i],:])
                    kept_scores.append(normalized_scores[indices[i]])
                    kept_indices.append(indices[i])
                    j=i

            figure =plt.figure()
            # Create the scatter plot
            scatter = plt.scatter(np.arange(n), normalized_scores, s=100, c=new_points[:, -1])

            plt.scatter(kept_indices, kept_scores,s=50,c = 'k',marker="*")
            legend1 = plt.legend(*scatter.legend_elements(), title="Names", loc='upper left')
            # legend2 = plt.legend([colors])
            plt.gca().add_artist(legend1)
            plt.savefig(self.OutputFolder+'graphs/data_recommended_rnd_{}.pdf'.format(rnd))
            plt.clf()
            # plt.show()
        else:
            kept_points = self.read_initial_recommended(rnd)

         
        np.savetxt(self.OutputFolder+'data/data_recommended/rnd'+str(rnd)+'.txt',
                    kept_points,
                    fmt='%.12f',
                    header=self.header)
        
    def get_key(self,dictionary, value):
        for key, val in dictionary.items():
            if val == value:
                return key
        return None  # Value not found in dictionary
    
    def get_trained_points(self,rnd, points):
        # print('points shape',np.shape(points))
        data=np.genfromtxt(self.OutputFolder + 'data/data_sampled/CASMresults'+str(rnd)+'.txt',dtype=np.float64)
        
        mu_values = []
        derivative_dim = self.derivative_dim  # Use shorthand for better readability
        
        # Preprocess data into a dictionary for faster lookup
        kappa_dict = {tuple(np.round(row[:derivative_dim], decimals=6)): row[-derivative_dim:] for row in data}
        # print(points[0])
        # print(kappa_dict)

        def create_key(array, dim, precision=6):
            return tuple(np.round(array[:dim], decimals=precision))

        # Create the dictionary with consistent key creation
        kappa_dict = {create_key(kappa, derivative_dim): kappa[-derivative_dim:] for kappa in data}

        # for kappa in data:
        #     key = create_key(kappa, derivative_dim)
        #     # key = tuple(np.round(kappa[:derivative_dim], decimals=7))  # Optional: rounding for precision
        #     print('key',type(key))
        #     if key in kappa_dict:
        #         print('append',kappa_dict[key])

        i=0
        # I = np.ones((np.shape(points)[0]), dtype=int)
        I= np.full(np.shape(points)[0], True)

        for kappa in points:
            key = create_key(kappa, derivative_dim)
            
            if key in kappa_dict:
                # print('append',kappa_dict[key])
                # print('key',key)
                # print('mu_og',kappa_dict[key])
                mu_values.append(kappa_dict[key])
            else:
                # print('Key not found!',key)
                I[i]=False
                # print(I[i])
            i+=1

        I.astype(bool)
        
        return np.array(mu_values),I

    def diff_in_mu(self,mu0,mu1):
        mse = mean_squared_error(mu1,mu0)
        return mse


    def determine_improvement(self,rnd):
        points = self.read_recommended(rnd)
        mu_real,I = self.get_trained_points(rnd,points) 
        points = points[I,:]

        # predict with old model
        pred_new = self.model.predict([points[:,:self.dim],points[:,self.dim:]])
        
        #load old model and predict
        self.model.load_trained_model(rnd-1) #? should it be rnd-1 or rnd-2? 
        pred_old = self.model.predict([points[:,:self.dim],points[:,self.dim:]])
        self.model.load_trained_model(rnd) # should be rnd or rnd-1? 



        categories = points[:,-1]


        mse_old = self.diff_in_mu(pred_old[1],mu_real)
        mse_new = self.diff_in_mu(pred_new[1],mu_real)

        mu_old = pred_old[1]
        mu_new = pred_new[1]


        type_averages_new = {
        category: mean_squared_error(
            mu_new[categories == self.type_criterion[category]],
            mu_real[categories == self.type_criterion[category]]
        ) if (categories == self.type_criterion[category]).sum() > 0 else 0
        for category in self.type_criterion
        }

        type_averages_old = {
            category: mean_squared_error(
                mu_old[categories == self.type_criterion[category]],
                mu_real[categories == self.type_criterion[category]]
            ) if (categories == self.type_criterion[category]).sum() > 0 else 0
            for category in self.type_criterion
        }


        # Calculate the average improvement in MSE for each type
        type_improvements = {category: type_averages_old[category] - type_averages_new[category] for category in self.type_criterion}
        return type_improvements

    def stopping_criteria(self,rnd):
        input, output = self.model.load_data(rnd, True)
        input, output = self.model.input_columns_to_training(input, output)
        current_mse = self.model.model_evaluate(input, output)




        # Get loss of previous idnn
        keras.backend.clear_session()
        lastmodel = self.model.load_trained_model(rnd-1)
        prev_mse =self.model.model_evaluate(input, output)

        # Reload current IDNN
        self.model.load_trained_model(rnd)

        if isinstance(prev_mse, list):
            prev_mse = prev_mse[2]
            current_mse  = current_mse[2]




        if  prev_mse - current_mse < self.stopping_alpha*current_mse :
            return True
        
        return False


    def reweight_criterion(self, rnd, type_improvements):
        updated_indices = []  # List to track which weights are being updated

        for category in self.type_criterion:
            if category != 'QBC' and not (category == 'hessian' and self.non_convexities_points_found == False):
                print('category', category)
                improvement = type_improvements[category]
                print('improvement', improvement)

                if not isnan(improvement):
                    index = self.type_criterion[category]
                    self.weights[index] += self.alpha * improvement
                    # Ensure no negative weights
                    if self.weights[index] < 0:
                        self.weights[index] = 0
                    updated_indices.append(index)

        # Normalize only updated weights
        total_weight = sum(self.weights[index] for index in updated_indices)
        if total_weight > 0:  # Avoid division by zero
            for index in updated_indices:
                self.weights[index] = (self.weights[index] / total_weight) * len(updated_indices)

        # Save the updated weights
        file_path = self.OutputFolder + 'data/data_recommended/weights.txt'
        with open(file_path, 'a') as f:
            np.savetxt(f, np.hstack((rnd, self.weights)).reshape(1, -1), fmt='%.6f', delimiter=' ')
        print('self.weights', self.weights)
    
    
    def return_weights_default(self):
        self.weights = np.ones(len(self.type_criterion))

        



    def explore_extended(self,rnd,twoD=True):


        global_points = self.prediction_points
        if twoD:
            global_points_2d = round(self.prediction_points*self.points_2d)
        output = np.ones((global_points,1))
        outputorder = []


       
        for domain in self.sampling_dict['continuous_dependent'] :
            # outputorder += self.sampling_dict['continuous_dependent'][domain]['dim']*self.sampling_dict['continuous_dependent'][domain]['values']
            test_set = self.sampling_dict['continuous_dependent'][domain]['type']

            
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
                x_test,eta,self.seed = self.create_test_set_sobol(global_points,
                                                            self.sampling_dict['continuous_dependent'][domain]['dim'],
                                                            bounds=x_bounds,
                                                            seed=self.seed)

            # print(test_set)
            if test_set == 'billiardwalk':
        # sample quasi-uniformly
                N_b = round(global_points/4)
                n_planes = self.sampling_dict['continuous_dependent'][domain]['n_planes']
                c_planes = self.sampling_dict['continuous_dependent'][domain]['c_planes']
                self.x0, eta = self.create_test_set_billiardwalk(self.x0,n_planes,c_planes,global_points,N_boundary=N_b)
                print('Create billiard sample set')
                print('eta',np.shape(eta))
                if twoD and global_points_2d >1:
                    n_planes = self.sampling_dict['continuous_dependent'][domain]['n_planes_2d']
                    c_planes = self.sampling_dict['continuous_dependent'][domain]['c_planes_2d']
                    perturb = self.sampling_dict['continuous_dependent'][domain]['2d_perturb']
                    self.x0_2d, eta_2d = self.create_test_set_billiardwalk(self.x0_2d,n_planes,c_planes,global_points_2d,N_boundary=N_b)
                    otheretas= np.zeros((np.shape(eta_2d)[0],self.dim-2))
                    otheretas += perturb*(np.random.rand(*otheretas.shape)-.5)
                    eta_2d = np.hstack((eta_2d,otheretas))
                    I = np.shape(output)[0]- np.shape(eta_2d)[0]
                    eta= np.vstack((eta[:I,:],eta_2d))
                output = np.hstack((output,eta))
       
        for domain in self.sampling_dict['continuous_independent']:
            range = self.sampling_dict['continuous_independent'][domain]
            [dim] = self.dict.get_individual_keys(domain,['dimensions'])
            random_continous = np.random.uniform(low=range[0], high=range[1], size=(global_points,dim))
            output = np.hstack((output,random_continous))


        

        for domain in self.sampling_dict['discrete']:
            range = self.sampling_dict['discrete'][domain]
            [dim] = self.dict.get_individual_keys(domain,['dimensions'])
            random_discrete = np.random.choice(range,global_points*dim)
            random_discrete = np.reshape(random_discrete,(global_points,dim))
            output = np.hstack((output,random_discrete))
        
        output = output[:,1:]

        if os.path.isfile(self.OutputFolder+'data/data_recommended/global_prediction_points_rnd'+str(rnd-1)+'.txt'):
            os.remove(self.OutputFolder+'data/data_recommended/global_prediction_points_rnd'+str(rnd-1)+'.txt')

        np.savetxt(self.OutputFolder+'data/data_recommended/global_prediction_points_rnd'+str(rnd)+'.txt',output,fmt='%.12f',
                    header=self.header)

    def query_by_committeee(self,rnd,set_size):
        print('Query by committee rnd',rnd)
        data = np.loadtxt(self.OutputFolder+'data/data_recommended/global_prediction_points_rnd'+str(rnd)+'.txt')

        
        predictions = np.zeros((set_size,np.shape(data)[0],self.inputs_dim))
        # num = self.QBC_repeat



        for i in range(set_size):
            try:
                predictions[i,:,:] = np.loadtxt(self.OutputFolder + 'training/predictions/prediction_{}_{}.json'.format(rnd,i))
            except:
                print('Could not find ',self.OutputFolder + 'training/predictions/prediction_{}_{}.json'.format(rnd,i))

        print(np.shape(predictions))

        predictions = predictions[~np.all(predictions == 0, axis=(1, 2))]

        print(np.shape(predictions))

        pred_set_size = np.shape(predictions)[0]

        avg = np.zeros((np.shape(data)[0],self.inputs_dim))
        for i in range(pred_set_size):
            avg += predictions[i,:,:]
        avg = avg/pred_set_size

        flucuation=np.zeros((np.shape(data)[0],self.inputs_dim))
        for i in range(pred_set_size):
            flucuation += (predictions[i,:,:]- avg)**2
        flucuation = flucuation/pred_set_size

        # print(np.shape(predictions))

        # variations = np.std(flucuation, axis=1)

        variations = np.sum(flucuation,axis=1)

        # Get the indices of the data points with the largest variations
        indices = np.argsort(-variations)[::-1]
        
        data_sorted= data[indices,:]



        # num = round(num*self.weights[self.type_criterion['QBC']])
        # if num < 10:
        #     num = 10

        # data_sorted= data_sorted[:num,:]

        input = [data_sorted[:,0:self.dim],data_sorted[:,self.dim:]]
        input_local = self.find_new_points(input, self.QBC_repeat_points, self.QBC_repeat,[self.QBC_perturb,0.5], 'QBC')
        # print('inputs3',np.shape(input_local))
        input_local = self.combine_list(input_local)

        self.write(rnd+1, 'QBC', input_local)


    def predict_explore_extended(self,rnd):
        # print('Loading data...')
        self.inputs_explore= np.loadtxt(self.OutputFolder+'data/data_recommended/global_prediction_points_rnd'+str(rnd)+'.txt')
        print('Predicting...')
        self.output_explore = self.model.predict([self.inputs_explore[:,0:self.dim],self.inputs_explore[:,self.dim:]])



########################################
        ##exploit hessian values
    def non_convexities(self,rnd,tol=0.1):
        print('Finding nonconvexities rnd',rnd )
        inputs = self.inputs_explore.copy()
        eigen = self.eigenvalues.copy()
        def check_negative_eigenvalues(eigenvalues):
        #Check if any of the eigenvalues for each point is less than zero
            # return np.any(eigenvalues < 0, axis=1)
            return eigenvalues[:,0] < .01

        # print('eigenvalues',eigen[0:5,:])

        I = check_negative_eigenvalues(eigen)
        # print('I', I[0:5])
        # print('shape of nonconvexities unfiltered',np.shape(inputs))
        inputs = inputs[I,:]
        # print('shape of nonconvexities filtered',np.shape(inputs))
        input = [inputs[:,0:self.dim],inputs[:,self.dim:]]
        input_local = self.find_new_points(input, self.non_convexities_repeat_points, self.non_convexities_repeat,[self.non_convexities_perturb,0.5], 'non_convexities')
        # print('inputs3',np.shape(input_local))
        # print('non_convexitites energy', input_local)
        input_local = self.combine_list(input_local)

        # print('inputs4',np.shape(input_local))

        if np.shape(input_local)[0] != 0:
            self.non_convexities_points_found=True
            self.write(rnd, 'non_convexities', input_local)
        else:
            self.non_convexities_points_found=False
            print('No nonconvex data points found')
        return input_local
    ########################################


    def sensitivity(self,rnd):
        print('Finding sensitive regions rnd',rnd)
        inputs = self.inputs_explore.copy()
        eigenvalues = self.eigenvalues.copy()
        sensitivity_score = np.max(eigenvalues, axis=1)
        sorted_indices = np.argsort(sensitivity_score)[::-1]
        sorted_inputs = inputs[sorted_indices]
        input = [inputs[:,0:self.dim],inputs[:,self.dim:]]
        # print('sensitivity',self.sensitivity_repeat_points,self.sensitivity_repeat)
        # print('input',input)
        input_local = self.find_new_points(input, self.sensitivity_repeat_points, self.sensitivity_repeat,[self.sensitivity_perturb,0.5], 'sensitivity')
        # print('input_local',input_local)
        input_local = self.combine_list(input_local)
        self.write(rnd, 'sensitivity', input_local)
        # return input_local
        # sorted_sensitivity_scores = sensitivity_score[sorted_indices]
        
        # return sorted_inputs, sorted_sensitivity_scores