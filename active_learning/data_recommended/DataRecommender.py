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

class DataRecommender():

    def __init__(self,model,dictionary): 
        self.dict = dictionary
        self.model = model
        [self.N_global_pts, self.sample_known_wells,self.wells,
         self.wells_points,self.sample_known_vertice,self.vertices, 
         self.vertice_points] = self.dict.get_category_values('Explore_Parameters')
        # [self.inputs_alias,self.OutputFolder] = self.dict.get_individual_keys(['input_alias','OutputFolder'])
        [self.sample_non_convexities,self.non_convexities_repeat, self.non_convexities_repeat_points,self.sample_high_error,
        self.high_error_repeat, self.high_error_repeat_points, self.sample_find_wells, self.wells_repeat,
        self.wells_repeat_points, self.lowest_free_energy,self.lowest_repeat,self.lowest_free_energy_file,
        self.sample_sensitivity,self.sensitivity_repeat_points, self.sensitivity_repeat,self.QBC,self.QBC_repeat] = self.dict.get_category_values('Exploit_Parameters')
        self.header = ''

        [self.Model_type,    
         self.Data_Generation, self.Data_Generation_Source, self.restart,
         self.inputs_data,self.inputs_alias,self.output_alias, self.iterations,
         self.OutputFolder, self.seed, self.inputs_dim, self.derivative_dim,
         self.output_dim,self.config_path,self.T,self.graph,self.reweight,self.alpha,self.prediction_points] = self.dict.get_category_values('Main')

        self.sampling_dict = self.dict.get_category('Sampling')
        self.Qinv = self.sampling_dict['continuous_dependent']['etasampling']['invQ']
        self.min_param=.001
        self.type_criterion= {}
        if os.path.isfile(self.OutputFolder+'data/data_recommended/types.txt'):
            with open(self.OutputFolder+'data/data_recommended/types.txt') as f: 
                data = f.read() 
            self.type_criterion = json.loads(data)
        # self.alpha=100000

        self.dim=self.inputs_dim-1

    def keep_good_output(self,output,type):
        eta = output[:,0:self.dim]
        # self.sampling_dict['continuous_dependent'][domain]['type']
        # Qinv = self.sampling_dict['continuous_dependent']['etasampling']['invQ'] 

        keep = np.ones((np.shape(eta)[0]), dtype=np.int8)
        # count=0
        # count2=0

        for k in range(np.shape(eta)[0]):
            isbad = False 
            for i in range(31):
                value = 0
                for j in range(6):
                    value += self.Qinv[i,j]*eta[k,j]
                # print('value',value)
                if value > 1.1 or value < -0.1:
                    # count+=1
                    # keep[k]=0
                    isbad = True
            # if isbad == False:
            #     # print(keep[k])
            #     count2+=1
            #     keep[k]= 1
            # else:
            #     keep[k]=0
                # print(keep[k])
            # if isbad == True:
            #     # print(keep[k])
            #     keep[k]=0
            #     # print(keep[k])
        # keep = np.array(keep)
        true_false_array = [bool(x) for x in keep]
        return  output[true_false_array,:]
                
        
    def create_types(self,type):   
        # print(self.type_criterion)
        self.type_criterion[type] = len(self.type_criterion)

        self.weights = np.ones(len(self.type_criterion))
        with open(self.OutputFolder+'data/data_recommended/types.txt', 'w') as file: 
            file.write(json.dumps(self.type_criterion))

    def write(self,rnd,type,output):
        value=self.type_criterion[type]
            

        output = np.hstack((output,self.type_criterion[type]*np.ones((np.shape(output)[0],1))))


        np.savetxt(self.OutputFolder+'data/data_recommended/'+type+'_rnd'+str(rnd)+'.txt',output,fmt='%.12f',
                    header=self.header)

        if os.path.isfile(self.OutputFolder+'data/data_recommended/initial_rnd'+str(rnd)+'.txt'):
            allResults = np.loadtxt(self.OutputFolder+'data/data_recommended/initial_rnd'+str(rnd)+'.txt')
            output = np.vstack((allResults,output))
        np.savetxt(self.OutputFolder+'data/data_recommended/initial_rnd'+str(rnd)+'.txt',
                    output,
                    fmt='%.12f',
                    header=self.header)


    def load_data(self,rnd,singleRnd=True):
        print('loading data')
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
        # print('input_local ie return',np.shape(input_local))
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
        
        input_local = self.find_new_points(input,self.wells_repeat_points,self.wells_repeat,[0.15,.5],'find_wells')
        input_local = self.combine_list(input_local)

        if np.shape(input_local)[0] != 0:
            self.write(rnd, 'find_wells', input_local)
        else:
            print('No data points near wells')


        return input_local
    

    def sample_external_data(self,rnd,path,value, name):
        columns = np.load(path,allow_pickle=True) 
        print('wells',columns)
        #EDIT - be more generic
        

        columns = [columns[:,0:self.dim],columns[:,self.dim:]]
        input_local = self.find_new_points(columns,[10],[self.wells_points],[value[0],value[1]])
        input_local= self.combine_list(input_local)
        self.write(rnd, name, input_local)


    def sample_wells(self, rnd):
        # self.sample_external_data(rnd,self.wells,[.15,0.5],'sample_wells')
        etaW = np.zeros((2*self.dim,self.dim))
        T = self.T*np.ones((2*self.dim,1))
        # wells
        etaW[:,0] = 0.5
        for i in range(1,self.dim):
            etaW[2*i,i] = 0.425
            etaW[2*i+1,i] = -0.425
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
        # print('well points',self.wells_points)
        # print('weight',self.weights[self.type_criterion['sample_wells']])
        N_w = round(self.wells_points*self.weights[self.type_criterion['sample_wells']])  #35 #50
        if N_w < 10:
            N_w=10
        # print('N_w',N_w)
        kappaW = np.repeat(kappaW,N_w,axis=0)
        kappaW  += .1*(np.random.rand(*kappaW.shape)-0.5)
        temp = self.T*np.ones((2*self.dim*N_w,1))
        input_local = np.hstack((kappaW,temp))
        # print('size of sample wells',np.shape(input_local))
        self.write(rnd, 'sample_wells', input_local)
           
    def sample_vertices(self,rnd):
        # self.sample_external_data(rnd,self.vertices,[-.05,0],'sample_vertices')
        etaB = np.zeros((2*(self.dim-1),self.dim))
        T = self.T*np.ones((2*(self.dim-1),1))
        # wells
        etaB[:,0] = 0.5
        for i in range(1,self.dim):
            etaB[2*i-2,i] = 0.5
            etaB[2*i-1,i] = -0.5
        if rnd<100:
            kappaB = etaB
        else:
            phi = np.array([10.0,0.1,0.1,0.1,0.1,0.1,0.1])
            muB = self.model.predict([etaB,T])[1]
            kappaB = etaB + 0.5*muB/phi

        N_w2 = round(self.vertice_points*self.weights[self.type_criterion['sample_vertices']]) # Number of random points per vertex
        if N_w2<10:
            N_w2=10
        kappaW2 = np.zeros((2*(self.dim-1)*N_w2,self.dim))
        temp = self.T*np.ones((2*(self.dim-1)*N_w2,1))
        kappaW2[:,0] = kappaB[0,0]
        kappaW2 += 0.05*(np.random.rand(*kappaW2.shape)-0.5) # Small random perterbation
        for i in range(1,self.dim):
            for j in range(2*N_w2):
                kappaW2[2*(i-1)*N_w2 + j,i] = np.random.rand()*(kappaB[2*i-2,i] - kappaB[2*i-1,i]) + kappaB[2*i-1,i] # Random between positive and negative well
        input_local = np.hstack((kappaW2,temp))
        self.write(rnd, 'sample_vertices', input_local)
    

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
    
    def construct_input_types(self):
        [self.type_criterion_of_input] = self.dict.get_individual_keys('Ordering',['type_of_input'])


    def explore(self,rnd):

        global_points = round(self.N_global_pts*self.weights[self.type_criterion['billiardwalk']]) 
        if global_points < 10:
            global_points=10
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
                if rnd<6:
                    N_b = round(global_points/4)
                else:
                    N_b = 0
                print('Create billiard sample set')
                n_planes = self.sampling_dict['continuous_dependent'][domain]['n_planes']
                c_planes = self.sampling_dict['continuous_dependent'][domain]['c_planes']
                x_test, eta = self.create_test_set_billiardwalk(
                    self.sampling_dict['continuous_dependent'][domain]['x0'],
                    n_planes,c_planes,global_points,
                                        N_boundary=N_b)
            output = np.hstack((output,eta[0:global_points,:]))
            # self.type_criterion_of_input = np.hstack((self.type_criterion_of_input,np.zeros(1)))
        for domain in self.sampling_dict['continuous_independent']:
            # outputorder += self.sampling_dict['continuous_independent'][domain]['dim']*self.sampling_dict['continuous_independent'][domain]['values']
            range = self.sampling_dict['continuous_independent'][domain]
            [dim] = self.dict.get_individual_keys(domain,['dimensions'])
            random_continous = np.random.uniform(low=range[0], high=range[1], size=(global_points,dim))
            output = np.hstack((output,random_continous))


        

        for domain in self.sampling_dict['discrete']:
            # print(self.sampling_dict['discrete'][domain])
            # outputorder += self.sampling_dict['discrete'][domain]['dim']*self.sampling_dict['discrete'][domain]['values']
            range = self.sampling_dict['discrete'][domain]
            [dim] = self.dict.get_individual_keys(domain,['dimensions'])
            random_discrete = np.random.choice(range,global_points*dim)
            random_discrete = np.reshape(random_discrete,(global_points,dim))
            output = np.hstack((output,random_discrete))
        
    #EDIT - reorder these to match input_alias 

        # print(self.type_criterion_of_input)
        output = output[:,1:]
         ## EDIT - temporary 
        output = output[0.5-np.abs(output[:,0]-0.5) > np.abs(output[:,1]) ]



        self.write(rnd, test_set, output)       


    def relevent_columns(self,inputs):
        return np.hstack((inputs[0],inputs[1]))


    def find_eigenvalues_explore(self,rnd):
        inputs = self.inputs_explore.copy()
        # output = self.output.copy()
        pred = self.output_explore.copy()
        free = pred[0]
        mu = pred[1]
        hessian= pred[2]
        eigen = np.zeros(mu.shape)
        eigenvector = np.zeros(hessian.shape)

        # print(np.shape(eigen))

        for i in range(len(hessian)):
            eigen[i,:], eigenvector[i,:,:] = LA.eig(hessian[i,:,:])
        
        self.eigenvalues = eigen

   

    def get_latest_pred(self,rnd):
        # print('Loading data...')
        self.inputs,self.output = self.load_data(rnd-1)
        print('Predicting...')
        self.output_pred = self.model.predict(self.inputs.copy())
        # print('latest prediction',self.output_pred )
    


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
            
            # print('high error og points order',inputs)
            input_local = self.find_new_points(inputs, self.high_error_repeat_points, self.high_error_repeat,[.01,0.5], 'high_error')
   
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
        # n = [50,50,25,10,0,0,0,0,0]
        # for i in range(self.dim):
        #     n[i]=int(n[i]*self.weights[self.type_criterion['lowest_free_energy']])

    

        # eta0 = np.linspace(0.45,0.55, n[0])
        # eta1 = np.linspace(0,0.5,n[1])
        # eta2 = np.linspace(0,0.25,n[2])
        # eta3 = np.linspace(0,0.16,n[3])
        
        # # print('created etas')
        # eta = np.meshgrid(eta0,eta1,eta2,eta3)
        # # print('created meshgrid')
        # etainput = np.array([eta[i].flatten()  for i in range(4)]).T

        # etainput = etainput[etainput[:,1]>etainput[:,2]]
        # etainput = etainput[etainput[:,2]>etainput[:,3]]
        # # eta = etainput[self.isindomain(etainput)]
        # additional_etas = np.zeros((np.shape(etainput)[0],3))
        # print(np.shape(etainput))
        # print(np.shape(additional_etas))
        # eta = np.hstack((etainput,additional_etas))

        # I = self.isindomain(eta[:,0:self.dim])
        # # print('I',I)
        # eta = eta[I,:]

        # np.savetxt('eta_curve_zigzag.txt',eta)

        eta = np.loadtxt(self.lowest_free_energy_file)
        # eta=np.loadtxt('/expanse/lustre/scratch/jholber/temp_project/git/row/active-learning/tests/LCO_row/eta_curve_2.txt')
        T= self.T*np.ones((np.shape(eta)[0],1))
        pred_new = self.model.predict([eta,T])
        free_new = pred_new[0]
        mu_new = pred_new[1]
        # print('pred',pred_new)

        data = np.hstack((eta,mu_new,free_new))

        data_list ={'x': data[:,0],'free': free_new[:,0]}
        resultlist = ['x']
        
        for i in range(1,self.dim):
            resultlist.append('eta_{}'.format(i))
            data_list['eta_{}'.format(i)] = data[:,i]


        df = pd.DataFrame(data_list)

        # Group by x1 and find the indices of the minimum y values within each group
        idx = df.groupby('x')['free'].idxmin()


        # print('resultlist',resultlist)
        # Get the corresponding x2, x3, and y values for the indices
        result = df.loc[idx, resultlist]
        # print(result[0:2,:])
        temp = self.T*np.ones((np.shape(result)[0],1))
        input_local = np.hstack((result,temp))
        # print(input_local[0:2,:])
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
        # print('shortest_distance',np.min(shortest_distances))
        # print('average of shortest_distance',np.average(shortest_distances))
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

    def choose_points(self,rnd):
        kept_points=[]
        if rnd > 0:
            print('Begin Reduction of Points, round ',rnd,'...')
            new_points = self.read_initial_recommended(rnd)
            old_points = self.read_recommended(0)
            for i in range(1,rnd):
                old_points=np.vstack((old_points,self.read_recommended(i)))
            novelty_scores = self.calculate_novelty_score_distance(new_points[:,:-1],old_points[:,:-1])
            normalized_scores = novelty_scores / np.sum(novelty_scores)
            n = len(novelty_scores)

            import matplotlib.pyplot as plt
            # print('np.average(normalized_scores)',np.average(normalized_scores))
            # plt.scatter(np.arange(n), normalized_scores)
            # plt.show()

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
                compare = np.random.rand()*np.average(normalized_scores)
                # print('compare',compare,'versus current',curr_score)
                # print(curr_score)
                if (curr_score > prev_score or curr_score > compare) and curr_score != 0:
                    kept_points.append(new_points[indices[i],:])
                    kept_scores.append(normalized_scores[indices[i]])
                    kept_indices.append(indices[i])
                    j=i
            # print('original num',n)
            # print('kept num',len(kept_scores))
            # print('max normalized scores',np.max(normalized_scores))
            # print('max kept scores',np.max(kept_scores))

            # print(self.type_criterion)
            
            # print('np.aranage(n)',np.shape(np.arange(n)))
            # print('normalized scores', np.shape(normalized_scores))
            # print('color',np.shape(new_points[:,-1]))

            # value_to_color = {
            #     1: 'red',
            #     2: 'blue',
            #     3: 'green',
            #     4: 'purple',
            #     5: 'orange',
            #     6: 'grey'
            # }

            # # Create a list of colors based on the values in new_points[:, -1]
            # colors = [value_to_color[val+1] for val in new_points[:, -1]]


            figure =plt.figure()
            # Create the scatter plot
            scatter = plt.scatter(np.arange(n), normalized_scores, s=100, c=new_points[:, -1])

            # Create a colormap
            # cmap = plt.cm.get_cmap('viridis', 6)  # Change 'viridis' to any colormap you prefer

            # Map values to colors using the colormap
            # colors = cmap(new_points[:, -1] / 5)  # Normalize to [0, 1] for the colormap



            # scatter=plt.scatter(np.arange(n), normalized_scores,c=new_points[:,-1])
            # seen = set()
            # values = [x for x in new_points[:,-1] if not (x in seen or seen.add(x))]
            # colors = [self.get_key(self.type_criterion,val) for val in values]
            # plt.scatter(kept_indices, kept_scores,c = new_points[kept_indices,-1],marker="*")
            plt.scatter(kept_indices, kept_scores,s=50,c = 'k',marker="*")
            legend1 = plt.legend(*scatter.legend_elements(), title="Names", loc='upper left')
            # legend2 = plt.legend([colors])
            plt.gca().add_artist(legend1)
            plt.savefig(self.OutputFolder+'graphs/data_recommended_rnd_{}.pdf'.format(rnd))
            plt.clf()
            # plt.show()
        else:
            kept_points = self.read_initial_recommended(rnd)

        # print('rnd ',rnd)
        # print('kept_points',kept_points[0:5,:])
         
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
        data=np.genfromtxt(self.OutputFolder + 'data/data_sampled/CASMresults'+str(rnd-1)+'.txt',dtype=np.float32)
        mu_values = []
        for kappa in points:
            # Find the row in the data where the kappa matches
            for row in data:
                if np.allclose(row[:self.derivative_dim], kappa[:self.derivative_dim]):
                    mu_values.append(row[-self.derivative_dim:])
                    break
        return np.array(mu_values)
        #find points in inputs
        #return outputs 

    def diff_in_mu(self,mu0,mu1):
        mse = np.linalg.norm(mu1 - mu0, axis=1)
        return mse



    def reweight_criterion(self, rnd):
        points = self.read_recommended(rnd-1)
        
        # predict with old model
        pred_new = self.model.predict([points[:,:self.dim],points[:,self.dim:]])
        
        #load old model and predict
        self.model.load_trained_model(rnd-2) #? should it be rnd-1 or rnd-2? 
        pred_old = self.model.predict([points[:,:self.dim],points[:,self.dim:]])
        self.model.load_trained_model(rnd-1) # should be rnd or rnd-1? 

        #find actual values of mse 
        real_data = self.get_trained_points(rnd,points) 

        #current weights

        categories = points[:,-1]




        mse_old = self.diff_in_mu(pred_old[1],real_data)
        mse_new = self.diff_in_mu(pred_new[1],real_data)

        type_averages_new = {category: np.mean(mse_new[categories == self.type_criterion[category]]) for category in self.type_criterion}
        type_averages_old = {category: np.mean(mse_old[categories == self.type_criterion[category]]) for category in self.type_criterion}


        # Calculate the average improvement in MSE for each type
        type_improvements = {category: type_averages_old[category] - type_averages_new[category] for category in self.type_criterion}
        # print('self.weights',self.weights)
        for category in self.type_criterion:
            if category != 'QBC' and not (category == 'hessian' and self.non_convexities_points_found==False):
                # print('category',category)
                # print('category type',self.type_criterion[category])
                improvement = type_improvements[category]
                # print('improvement', improvement)

                if not isnan(improvement):
                    self.weights[self.type_criterion[category]] += self.alpha * improvement
                # if improvement > 0:
                #     # Increase the weight for this type if there's improvement
                #     self.weights[self.type_criterion[category]] += self.alpha * improvement
                # else:
                #     # Decrease the weight for this type if there's no improvement
                #     self.weights[self.type_criterion[category]] -= self.alpha * abs(improvement)
                if self.weights[self.type_criterion[category]] <0:
                    self.weights[self.type_criterion[category]]=0
                

    # Normalize weights to ensure they sum up to 1
        total_weight = sum(self.weights)
        self.weights = (self.weights/total_weight)*len(self.weights)
        print('self.weights',self.weights)
        # self.weight={category: weight / total_weight for category, weight in self.weights.items()}
    
    def return_weights_default(self):
        self.weights = np.ones(len(self.type_criterion))

        



    def explore_extended(self,rnd):


        global_points = self.prediction_points
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
                if rnd<6:
                    N_b = round(global_points/4)
                else:
                    N_b = 0
                print('Create billiard sample set')
                n_planes = self.sampling_dict['continuous_dependent'][domain]['n_planes']
                c_planes = self.sampling_dict['continuous_dependent'][domain]['c_planes']
                x_test, eta = self.create_test_set_billiardwalk(
                    self.sampling_dict['continuous_dependent'][domain]['x0'],
                    n_planes,c_planes,global_points,
                                        N_boundary=N_b)
            output = np.hstack((output,eta[0:global_points,:]))
            # self.type_criterion_of_input = np.hstack((self.type_criterion_of_input,np.zeros(1)))
        for domain in self.sampling_dict['continuous_independent']:
            # outputorder += self.sampling_dict['continuous_independent'][domain]['dim']*self.sampling_dict['continuous_independent'][domain]['values']
            range = self.sampling_dict['continuous_independent'][domain]
            [dim] = self.dict.get_individual_keys(domain,['dimensions'])
            random_continous = np.random.uniform(low=range[0], high=range[1], size=(global_points,dim))
            output = np.hstack((output,random_continous))


        

        for domain in self.sampling_dict['discrete']:
            # print(self.sampling_dict['discrete'][domain])
            # outputorder += self.sampling_dict['discrete'][domain]['dim']*self.sampling_dict['discrete'][domain]['values']
            range = self.sampling_dict['discrete'][domain]
            [dim] = self.dict.get_individual_keys(domain,['dimensions'])
            random_discrete = np.random.choice(range,global_points*dim)
            random_discrete = np.reshape(random_discrete,(global_points,dim))
            output = np.hstack((output,random_discrete))
        
    #EDIT - reorder these to match input_alias 

        # print(self.type_criterion_of_input)
        output = output[:,1:]
         ## EDIT - temporary 
        output = output[0.5-np.abs(output[:,0]-0.5) > np.abs(output[:,1]) ]



        # output = np.hstack((output,self.type_criterion[type]*np.ones((np.shape(output)[0],1))))

        np.savetxt(self.OutputFolder+'data/data_recommended/global_prediction_points_rnd'+str(rnd)+'.txt',output,fmt='%.12f',
                    header=self.header)
    

    def query_by_committeee(self,rnd,set_size):
        print('Query by committee rnd',rnd)
        data = np.loadtxt(self.OutputFolder+'data/data_recommended/global_prediction_points_rnd'+str(rnd)+'.txt')

        
        predictions = np.zeros((set_size,np.shape(data)[0],self.inputs_dim))
        num = self.QBC_repeat



        for i in range(set_size):
            predictions[i,:,:] = np.loadtxt(self.OutputFolder + 'training/prediction_{}_{}.json'.format(rnd,i))

        avg = np.zeros((np.shape(data)[0],self.inputs_dim))
        for i in range(set_size):
            avg += predictions[i,:,:]
        avg = avg/set_size

        flucuation=np.zeros((np.shape(data)[0],self.inputs_dim))
        for i in range(set_size):
            flucuation += (predictions[i,:,:]- avg)**2
        flucuation = flucuation/set_size

        # print(np.shape(predictions))

        variations = np.std(flucuation, axis=1)

        # Get the indices of the data points with the largest variations
        indices = np.argsort(-variations)
        
        data_sorted= data[indices,:]



        num = round(num*self.weights[self.type_criterion['QBC']])
        if num < 10:
            num = 10

        data_sorted= data_sorted[:num,:]

        self.write(rnd, 'QBC', data_sorted)


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
            return np.any(eigenvalues < 0, axis=1)

        # print('eigenvalues',eigen[0:5,:])

        I = check_negative_eigenvalues(eigen)
        # print('I', I[0:5])
        print('shape of nonconvexities unfiltered',np.shape(inputs))
        inputs = inputs[I,:]
        print('shape of nonconvexities filtered',np.shape(inputs))
        input = [inputs[:,0:self.dim],inputs[:,self.dim:]]
        input_local = self.find_new_points(input, self.non_convexities_repeat_points, self.non_convexities_repeat,[.04,0.5], 'non_convexities')
        # print('inputs3',np.shape(input_local))
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
        input_local = self.find_new_points(input, self.sensitivity_repeat_points, self.sensitivity_repeat,[.04,0.5], 'sensitivity')
        input_local = self.combine_list(input_local)
        self.write(rnd, 'sensitivity', input_local)
        # return input_local
        # sorted_sensitivity_scores = sensitivity_score[sorted_indices]
        
        # return sorted_inputs, sorted_sensitivity_scores