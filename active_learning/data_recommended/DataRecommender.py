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

class DataRecommender():

    def __init__(self,model,dictionary): 
        self.dict = dictionary
        self.model = model
        [self.N_global_pts, self.sample_known_wells,self.wells,
         self.wells_points,self.sample_known_vertice,self.vertices, 
         self.vertice_points] = self.dict.get_category_values('Explore_Parameters')
        # [self.inputs_alias,self.OutputFolder] = self.dict.get_individual_keys(['input_alias','OutputFolder'])
        [self.sample_hessian,self.hessian_repeat, self.hessian_repeat_points,
         self.sample_high_error, self.high_error_repeat, self.high_error_repeat_points,
         self.exploit_find_wells, self.wells_repeat,self.wells_repeat_points] = self.dict.get_category_values('Exploit_Parameters')
        self.header = ''

        [self.Model_type,    
         self.Data_Generation, self.Data_Generation_Source, self.restart,
         self.inputs_data,self.inputs_alias,self.output_alias, self.iterations,
         self.OutputFolder, self.seed, self.inputs_dim, self.derivative_dim,
         self.output_dim,self.config_path] = self.dict.get_category_values('Main')

        self.sampling_dict = self.dict.get_category('Sampling')
        self.Qinv = self.sampling_dict['continuous_dependent']['etasampling']['invQ']

    def keep_good_output(self,output,type):
        eta = output[:,0:7]
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
        # print('count',count)
        # print("count2",count2)

        # returnoutput = output[true_false_array,:]
        # print(np.shape(returnoutput))

        
        # np.savetxt('{}_keep.txt'.format(type),true_false_array)
        return  output[true_false_array,:]
                
        


    def write(self,rnd,type,output):
        # print('rnd',rnd,'type',type)
        # print('output',np.shape(output))
        # output = self.keep_good_output(output,type)
        # print('output',np.shape(output))




        np.savetxt(self.OutputFolder+'data/data_recommended/'+type+'_rnd'+str(rnd)+'.txt',output,fmt='%.12f',
                    header=self.header)
        
        # if os.path.isfile(self.OutputFolder+'data/data_recommended/'+type+'_rnd'+str(rnd)+'.txt'):
        #     allResults = np.loadtxt(self.OutputFolder+'data/data_recommended/'+type+'_rnd'+str(rnd)+'.txt')
        #     output = np.vstack((allResults,output))
        # np.savetxt(self.OutputFolder+'data/data_recommended/'+type+'_rnd'+str(rnd)+'.txt',
        #             output,
        #             fmt='%.12f',
        #             header=self.header)


        if os.path.isfile(self.OutputFolder+'data/data_recommended/rnd'+str(rnd)+'.txt'):
            allResults = np.loadtxt(self.OutputFolder+'data/data_recommended/rnd'+str(rnd)+'.txt')
            output = np.vstack((allResults,output))
        np.savetxt(self.OutputFolder+'data/data_recommended/rnd'+str(rnd)+'.txt',
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
        # inputs,output = self.model.input_columns_to_training(inputs,output,unique_inputs=False)

        # output = []

        return inputs,output
    

    #given a set of points, should perturb them if applicable
    #then write them
    #input is a list, where each entry is an array corresponding to 
    #different input variables. The array is ordered by importance
    #repeat is an array with the number of search points
    #duplicate is an array of same size of repeat, saying how many times
    #each point should be duplicated

    def find_new_points(self,inputs,repeat,duplicate,perturb):
        input_local = []
        for k in range(np.size(self.type_of_input)):
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

            if self.type_of_input[k] == 0 or self.type_of_input[k] == 1: 
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


    def find_wells(self,rnd,dim=4,bounds=[0,0.25],rereference=True):

    # Find "wells" (regions of convexity, with low gradient norm)
        

    # First, rereference the free energy
        inputs = self.inputs.copy()
        output = self.output.copy()
        pred = self.output_pred.copy()

        # input = [input[1], input[3]]

        mu = pred[1]
        # if rereference:
        #     eta_test = np.array([bounds[0]*np.ones(dim),
        #                         bounds[1]*np.ones(dim)])
        #     if self.model.unique_inputs:
        #         y = 0.01*self.model.predict([eta_test,eta_test,eta_test,T])[0]
        #     else:
        #         y = 0.01*self.model.predict(eta_test)[0]
        #     g0 = y[0,0]
        #     g1 = y[1,0]
        #     mu_test[:,0] = mu_test[:,0] - 1./bounds[1]*(g1 - g0)
        
        gradNorm = np.sqrt(np.sum(mu**2,axis=-1))

        H = pred[2] # get the list of Hessian matrices
        ind2 = convexMult(H) # indices of points with local convexity
        # eta = x[ind2]
        gradNorm = gradNorm[ind2]

        I = np.argsort(gradNorm)

        
        for k in range(np.size(self.type_of_input)):
            column = inputs[k]
            inputs[k] = column[I]
        
        input_local = self.find_new_points(inputs,self.wells_repeat_points,self.wells_repeat,[0.15,.5])
        input_local = self.combine_list(input_local)

        if np.shape(input_local)[0] != 0:
            self.write(rnd, 'find_wells', input_local)
        else:
            print('No data points in near wells')


        return input_local
    

    def sample_external_data(self,rnd,path,value, name):
        columns = np.load(path,allow_pickle=True) 
        # print('wells',columns)
        #EDIT - be more generic
        

        columns = [columns[:,0:7],columns[:,7:8]]
        # columns = [columns[:,0:1],columns[:,1:7],columns[:,7:8]]
        input_local = self.find_new_points(columns,[10],[self.wells_points],[value[0],value[1]])
        input_local= self.combine_list(input_local)
        self.write(rnd, name, input_local)


    def sample_wells(self, rnd):
        # self.sample_external_data(rnd,self.wells,[.15,0.5],'sample_wells')
        etaW = np.zeros((8,4))
        T = 260*np.ones((8,1))
        # wells
        etaW[:,0] = 0.5
        for i in range(1,4):
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

        N_w = self.wells_points #35 #50
        kappaW = np.repeat(kappaW,N_w,axis=0)
        kappaW  += .1*(np.random.rand(*kappaW.shape)-0.5)
        temp = 260*np.ones((8*N_w,1))
        input_local = np.hstack((kappaW,temp))
        self.write(rnd, 'sample_wells', input_local)
           
    def sample_vertices(self,rnd):
        etaB = np.zeros((2*(4-1),4))
        T = 260*np.ones((2*(4-1),1))
        # wells
        etaB[:,0] = 0.5
        for i in range(1,4):
            etaB[2*i-2,i] = 0.5
            etaB[2*i-1,i] = -0.5
        if rnd<100:
            kappaB = etaB
        else:
            phi = np.array([10.0,0.1,0.1,0.1,0.1,0.1,0.1])
            muB = self.model.predict([etaB,T])[1]
            kappaB = etaB + 0.5*muB/phi

        N_w2 = self.vertice_points # Number of random points per vertex
        kappaW2 = np.zeros((2*(4-1)*N_w2,4))
        temp = 260*np.ones((2*(4-1)*N_w2,1))
        kappaW2[:,0] = kappaB[0,0]
        kappaW2 += 0.05*(np.random.rand(*kappaW2.shape)-0.5) # Small random perterbation
        for i in range(1,4):
            for j in range(2*N_w2):
                kappaW2[2*(i-1)*N_w2 + j,i] = np.random.rand()*(kappaB[2*i-2,i] - kappaB[2*i-1,i]) + kappaB[2*i-1,i] # Random between positive and negative well
        input_local = np.hstack((kappaW2,temp))
        self.write(rnd, 'sample_vertices', input_local)
        # self.sample_external_data(rnd,self.vertices,[-.05,0],'sample_vertices')
    

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
        [self.type_of_input] = self.dict.get_individual_keys('Ordering',['type_of_input'])

    #     self.type_of_input = np.array([]) #type of input: 0 -continuous_dependent, 1 - continuos_independent, 2- discrete 
    #     self.model_order = np.array([])
    #     for i in range(np.size(self.inputs_alias)):
    #         domaintype = self.dict.get_individual_keys(self.inputs_alias[i],['domain_type'])
    #         if domaintype == 'continuous_dependent':
    #             self.type_of_input = np.hstack((self.type_of_input,np.zeros(1)))
    #         if domaintype == 'continuous_independent':
    #             self.type_of_input = np.hstack((self.type_of_input,np.ones(np.size(1))))
    #         if domaintype == 'discrete':
    #             self.type_of_input = np.hstack((self.type_of_input,2*np.ones(1)))
    #         modeltype = self.dict.get_individual_keys(self.inputs_alias[i],['derivative_dim'])
    #         if modeltype:
    #             self.model_order=np.hstack((self.model_order,np.ones(np.size(1))))
    #         else:
    #             self.model_order=np.hstack((self.model_order,np.zeros(np.size(1))))

    #     # for x in self.sampling_dict['continuous_dependent']:
    #     #     self.type_of_input = np.hstack((self.type_of_input,np.zeros(1)))
    #     # for y in self.sampling_dict['continuous_independent']:
    #     #     self.type_of_input = np.hstack((self.type_of_input,np.ones(np.size(1))))
    #     # for z in self.sampling_dict['discrete']:
    #     #     self.type_of_input = np.hstack((self.type_of_input,2*np.ones(1)))
    #     # #reorder

    def explore(self,rnd):

        output = np.ones((self.N_global_pts,1))
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
            # self.type_of_input = np.hstack((self.type_of_input,np.zeros(1)))
        for domain in self.sampling_dict['continuous_independent']:
            # outputorder += self.sampling_dict['continuous_independent'][domain]['dim']*self.sampling_dict['continuous_independent'][domain]['values']
            range = self.sampling_dict['continuous_independent'][domain]
            [dim] = self.dict.get_individual_keys(domain,['dimensions'])
            random_continous = np.random.uniform(low=range[0], high=range[1], size=(self.N_global_pts,dim))
            output = np.hstack((output,random_continous))


        

        for domain in self.sampling_dict['discrete']:
            # print(self.sampling_dict['discrete'][domain])
            # outputorder += self.sampling_dict['discrete'][domain]['dim']*self.sampling_dict['discrete'][domain]['values']
            range = self.sampling_dict['discrete'][domain]
            [dim] = self.dict.get_individual_keys(domain,['dimensions'])
            random_discrete = np.random.choice(range,self.N_global_pts*dim)
            random_discrete = np.reshape(random_discrete,(self.N_global_pts,dim))
            output = np.hstack((output,random_discrete))
        
    #EDIT - reorder these to match input_alias 

        # print(self.type_of_input)
        output = output[:,1:]
         ## EDIT - temporary 
        output = output[0.5-np.abs(output[:,0]-0.5) > np.abs(output[:,1]) ]



        self.write(rnd, test_set, output)       


    def relevent_columns(self,inputs):
        return np.hstack((inputs[0],inputs[1]))


   ########################################
        ##exploit hessian values
    def hessian(self,rnd,tol=0.1):
        inputs = self.inputs.copy()
        output = self.output.copy()
        pred = self.output_pred.copy()
        free = pred[0]
        mu = pred[1]
        hessian= pred[2]



        eigen = np.zeros(mu.shape)
        eigenvector = np.zeros(hessian.shape)

        # print(np.shape(eigen))

        for i in range(len(hessian)):
            eigen[i,:], eigenvector[i,:,:] = LA.eig(hessian[i,:,:])

        def arg_zero_eig(e,tol = 0.1):
            return  np.sum(np.abs(e) < tol,axis=1) == e.shape[1]


        

        # input = [input[1], input[3]]




        input_local = self.find_new_points(inputs, self.hessian_repeat_points, self.hessian_repeat,[.04,0.5])




        input_local = self.combine_list(input_local)

        if np.shape(input_local)[0] != 0:
            self.write(rnd, 'hessian', input_local)
        else:
            print('No data points in hessian tolerance')
        return input_local
    ########################################

    def get_latest_pred(self,rnd):
        # print('Loading data...')
        self.inputs,self.output = self.load_data(rnd-1)
        print('Predicting...')
        self.output_pred = self.model.predict(self.inputs.copy())
    


    def high_error(self,rnd):
        
        

        print('Finding high pointwise error...')

        inputs = self.inputs.copy()
        output = self.output.copy()
        output_pred = self.output_pred.copy()

        input_local = []

        print('inputs',inputs)
        print('outputs',inputs)

        for i in range(np.size(self.output_alias)):
            derivative,dimensions,adjust = self.dict.get_category_values(self.output_alias[i])
            # output[:][derivative] = (output[:][derivative]+adjust[0])*adjust[1]

            input_derivative = []


            error = np.sum((output_pred[derivative] - output[:][i])**2,axis=1)
            for k in range(np.size(self.type_of_input)):
                column = inputs[k]
                higherror =  column[np.argsort(error)[::-1],:]
                inputs[k]=higherror
            
            # print('high error og points order',inputs)
            input_local = self.find_new_points(inputs, self.high_error_repeat_points, self.high_error_repeat,[.004,0.5])
   
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
        n = [100,100,50,1]

        # eta0 = np.linspace(0.45,0.55, n[0])
        # eta1 = np.linspace(0,0.5,n[1])
        # eta2 = np.linspace(0,0.25,n[2])
        # eta3 = np.linspace(0,0.16,n[3])
        
        # print('created etas')
        # eta = np.meshgrid(eta0,eta1,eta2,eta3)
        # print('created meshgrid')
        # etainput = np.array([eta[i].flatten()  for i in range(4)]).T

        # etainput = etainput[etainput[:,1]>etainput[:,2]]
        # eta = etainput[self.isindomain(etainput)]

        # I = self.isindomain(etainput[:,0:4])
        # print('I',I)
        # eta = etainput[I,:]

        # np.savetxt('eta_curve_2.txt',eta)


        eta=np.loadtxt('/expanse/lustre/scratch/jholber/temp_project/git/row/active-learning/tests/LCO_row/eta_curve_2.txt')
        T= 260*np.ones((np.shape(eta)[0],1))
        pred_new = self.model.predict([eta,T])
        free_new = pred_new[0]
        mu_new = pred_new[1]

        data = np.hstack((eta,mu_new,free_new))

        data ={'x': data[:,0],'eta1': data[:,1],'eta2': data[:,2],'eta3': data[:,3],'free': free_new[:,0]}


        df = pd.DataFrame(data)

        # Group by x1 and find the indices of the minimum y values within each group
        idx = df.groupby('x')['free'].idxmin()

        # Get the corresponding x2, x3, and y values for the indices
        result = df.loc[idx, ['x', 'eta1', 'eta2', 'free']]
        # result.to_csv('lowest_free_energies_rnd{}.txt'.format(self.rnd), sep='\t', index=False)
        
        # kappa = np.repeat(eta,2,axis=0)
        # kappa[:,0]  += .1*(np.random.rand(*kappa[:,0].shape)-0.5)
        # kappa = eta
        temp = 260*np.ones((np.shape(result)[0],1))
        input_local = np.hstack((result,temp))
        self.write(rnd, 'lowest_free_energy', input_local)
        