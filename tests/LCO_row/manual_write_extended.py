import json
import numpy as np
from subprocess import check_output, STDOUT
import os, shutil, copy
import fileinput as fin
from time import sleep
from shutil import copyfile
# from active_learning.data_collector.sampling import Sampling
import pandas as pd
import sys

rnd = sys.argv[1]

def write(OutputFolder,rnd):
        # rows_to_keep = [0, 29, 30, 31]
        rows_to_keep = [0,1,2,3,4,5,6]
        kappa = []
        eta = []
        phi = []
        T = []
        # if not os.path.exists(f'round_{rnd}'):
        # dirname = OutputFolder + 'data/data_sampled/'.format(rnd)
        # os.mkdir(dirname)
        data_points = []
        # else:
        #     os.mkdir(f'round_{rnd}_{temp}')
        for dir in os.listdir(OutputFolder + 'data/data_sampled/round_{}'.format(rnd)):
            if 'job' in dir:
                if os.path.exists(OutputFolder + 'data/data_sampled/round_{}/'.format(rnd)+dir+'/results.json'):
                    with open(OutputFolder + 'data/data_sampled/round_{}/'.format(rnd)+dir+'/results.json','r') as file:
                        data = json.load(file) 
                        for i in range(len(data['<comp(a)>'])):
                            with open(OutputFolder + 'data/data_sampled/round_{}/'.format(rnd)+dir+ f'/conditions.{i}/conditions.json', 'r') as conditions_file:
                                conditions_data = json.load(conditions_file)
                                kappa = conditions_data['order_parameter_quad_pot_target']
                                phi = conditions_data['order_parameter_quad_pot_vector']
                                phi_subset=4*[0]
                                kappa_subset=4*[0]
                                mu = 4*[0]
                                eta = 4*[0]
                                # eta0 = c
                                k=0
                                for j in rows_to_keep:
                                    eta[k] = data['<order_parameter({})>'.format(j)][i]
                                    mu[k] =(-2*phi[j]*(eta[k]-kappa[j]))*32
                                    phi_subset[k] = phi[j]*32*32
                                    kappa_subset[k] = kappa[j]/32
                                    T = conditions_data['temperature']
                                    potential = data['<formation_energy>'][i]
                                    eta[k] = eta[k]/32
                                    k+=1
                                # print('kappa_subsest',kappa_subset)
                                # print('phi_subsest',phi_subset)
                                # print("mu",mu)
                                # print('t',T)
                                # print('eta',eta)
                                # sys.exit()
                                data_points.append({
                                    'kappa': kappa_subset,
                                    'phi': phi_subset,
                                    'mu': mu,
                                    'T': T,
                                    'eta': eta,
                                    'potential':potential
                                })


                    # shutil.move(OutputFolder + 'data/data_sampled/'+dir,dirname)

        kappa = np.array([d['kappa'] for d in data_points])
        eta = np.array([d['eta'] for d in data_points])
        phi = np.array([d['phi'] for d in data_points])
        T = np.array([d['T'] for d in data_points])
        mu= np.array([d['mu'] for d in data_points])
        potential= np.array([d['potential'] for d in data_points])
        T = np.reshape(T,(len(T),1))
        potential = np.reshape(potential,(len(potential),1))

        dataOut = np.hstack((kappa,eta,phi,T,mu,potential))
        dataOut = dataOut[~pd.isna(dataOut).any(axis=1)] #remove any rows with nan
        outVars = ['kappa','eta','phi']
        header = ''
        for outVar in outVars:
            for i in range(4):
                header += outVar+'_'+str(i)+' '
        header += 'T '
        for i in range(4):
            header += 'mu_'+str(i)+' '
        header += 'formation'

        # print("np.shape dataout",np.shape(dataOut))

        # print('dataOut',dataOut)

        np.savetxt(OutputFolder + 'data/data_sampled/CASMresults_extended{}.txt'.format(rnd),
                dataOut,
                header=header)
        if rnd==0:
            copyfile(OutputFolder + 'data/data_sampled/CASMresults_extended{}.txt'.format(rnd),OutputFolder + 'data/data_sampled/CASMallresults_extended{}.txt'.format(rnd))
        else:
            allResults = np.loadtxt(OutputFolder + 'data/data_sampled/CASMallresults_extended{}.txt'.format(rnd-1))
            allResults = np.vstack((allResults,dataOut))
            np.savetxt(OutputFolder + 'data/data_sampled/CASMallresults_extended{}.txt'.format(rnd),
                    allResults,
                    header=header)

        
        dataOut = np.hstack((eta,T,mu,potential))


        np.save(OutputFolder + 'data/data_sampled/results_extended{}'.format(rnd),
        dataOut)
        if rnd==0:
            np.save(OutputFolder + 'data/data_sampled/allresults_extended{}'.format(rnd),
                    dataOut)
        else:
            allResults =  np.load(OutputFolder + 'data/data_sampled/results_extended{}.npy'.format(rnd-1),allow_pickle=True)
            allResults = np.vstack((allResults,dataOut))

            np.save(OutputFolder + 'data/data_sampled/allresults_extended{}'.format(rnd),
                    allResults)
        
        np.savetxt(OutputFolder + 'data/data_sampled/results_extended{}.txt'.format(rnd),dataOut)
        


write('../../Output_2d/280_4/',int(rnd))