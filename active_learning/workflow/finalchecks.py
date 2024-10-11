import os
import sys, shutil
import numpy as np
import matplotlib.pyplot as plt
from active_learning.workflow.dictionary import Dictionary
from active_learning.model.idnn_model import IDNN_Model


def plotting_points(dictionary,model,outputfile,rnd_max):
    # Load data
    input, output = model.load_data_no_scale(rnd_max, False)

    etas = input[0]
    comp =etas[:,0]
    op_largest  = np.max(np.abs(etas[:,1:7]), axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(comp,op_largest, marker='o', linestyle='None', color='b')
    plt.xlabel('comp')
    plt.ylabel('orderparamters')
    plt.title('Points sampled, largest ordering vs composition')
    plt.grid(True)
    plt.savefig(outputfile+'data_sampled.png')
    plt.clf()


def loss(dictionary,model,outputfile,rnd_max):


    # Load data
    input, output = model.load_data(rnd_max, False)
    input, output = model.input_columns_to_training(input, output)

    file1 = 'allResults49.txt'
    eta = np.genfromtxt(file1,dtype=np.float32)[:,7:14]
    mu = np.genfromtxt(file1,dtype=np.float32)[:,22:]
    T= np.genfromtxt(file1,dtype=np.float32)[:,21:22]

    data = np.hstack((eta,T,mu))

    i,o = model.array_to_column(data)
    i,o = model.scale_loaded_data(i,o)
    i,o = model.input_columns_to_training(i, o)


    #graph points for surrogate model
    file2 = 'surrogate_model_full_rnd2.txt'
    eta = np.genfromtxt(file1,dtype=np.float32)[:,0:7]
    mu = np.genfromtxt(file1,dtype=np.float32)[:,7:14]
    T= np.genfromtxt(file1,dtype=np.float32)[:,14:15]

    data = np.hstack((eta,T,mu))

    g_input,g_output = model.array_to_column(data)
    g_input,g_output = model.scale_loaded_data(g_input,g_output)
    g_input,g_output = model.input_columns_to_training(g_input, g_output)


    # Initialize lists to store rnd values and corresponding losses
    rnd_values = []
    lastloss_values = []
    rndloss_values=[]
    allloss_values=[]
    graphpoints_values=[]

    # Loop through each rnd value
    for rnd in range(rnd_max+1):
        model.load_trained_model(rnd)
        # data = model.load_data(rnd_max)
        
        # Evaluate model and get loss
        print('evaluating')
        lastloss = model.model_evaluate(input, output)
        graphpointsloss = model.model_evaluate(g_input,g_output)
        allloss = model.model_evaluate(i,o)
        rndloss = model.loss(rnd,True)
        
        # Append rnd and loss values to lists
        rnd_values.append(rnd)
        lastloss_values.append(lastloss[2])
        rndloss_values.append(rndloss[2])
        allloss_values.append(allloss[2])
        graphpoints_values.append(graphpointsloss[2])
        
        print('rnd', rnd)
        print('loss', lastloss[2])
        print('loss', rndloss[2])
        print('loss', allloss[2])


    # Plot rnd vs loss
    plt.figure(figsize=(10, 6))
    plt.plot(rnd_values, lastloss_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Rnd')
    plt.ylabel('Loss')
    plt.title('Loss vs. Rnd for final data')
    plt.grid(True)
    plt.savefig(outputfile+'mse_finaldata.png')
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.plot(rnd_values, rndloss_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Rnd')
    plt.ylabel('Loss')
    plt.title('Loss vs. Rnd for rnd data')
    plt.grid(True)
    plt.savefig(outputfile+'mse_rnddata.png')
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.plot(rnd_values, allloss_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Rnd')
    plt.ylabel('Loss')
    plt.title('Loss vs. Rnd for allresultsdata')
    plt.grid(True)
    plt.savefig(outputfile+'mse_alldata.png')
    plt.clf()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(rnd_values, graphpoints_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Rnd')
    plt.ylabel('Loss')
    plt.title('Loss vs. Rnd for graph points data')
    plt.grid(True)
    plt.savefig(outputfile+'mse_graphdata.png')
    plt.clf()
    plt.show()

    np.savetxt(outputfile+'mse.txt',np.vstack((rnd_values,lastloss_values,rndloss_values,allloss_values,graphpoints_values)))



