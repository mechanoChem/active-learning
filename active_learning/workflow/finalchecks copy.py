import os
import sys, shutil
import numpy as np
import matplotlib.pyplot as plt
from active_learning.workflow.dictionary import Dictionary
from active_learning.model.idnn_model import IDNN_Model
from numpy import linalg as LA

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

def predicting_best(dictionary,model,outputfile,rnd):
    mse = np.loadtxt(outputfile+'mse.txt')
    last_loss = mse[1,:]
    lowestrnd  = np.argmin(last_loss)
    input, output = model.load_data_no_scale(rnd, False)
    eta = input[0]
    # input, output = model.input_columns_to_training(input, output)

    print('lowest rnd', lowestrnd)
    model.load_trained_model(lowestrnd)
    eta = input[0]
    T = input[1]
    pred = model.predict(input)
    free = pred[0]
    mu = pred[1]
    grad = pred[2]
    eigen = np.zeros(mu.shape)
    eigenvector = np.zeros(grad.shape)
    for i in range(len(grad)):
        eigen[i,:], eigenvector[i,:,:] = LA.eig(grad[i,:,:])
    print('max eta1', np.max(eta[:,1]))
    data = np.hstack((eta,mu,free,eigen,T))
    np.savetxt(outputfile+f'alldatapredictedrnd_{lowestrnd}.txt',data)

    file1 = 'allResults49.txt'
    eta = np.genfromtxt(file1,dtype=np.float64)[:,7:14]
    mu = np.genfromtxt(file1,dtype=np.float64)[:,22:]
    T= np.genfromtxt(file1,dtype=np.float64)[:,21:22]
    pred = model.predict([eta,T])
    free = pred[0]
    mu_pred = pred[1]
    grad = pred[2]
    eigen = np.zeros(mu.shape)
    eigenvector = np.zeros(grad.shape)
    for i in range(len(grad)):
        eigen[i,:], eigenvector[i,:,:] = LA.eig(grad[i,:,:])
    data = np.hstack((eta,mu_pred,free,eigen,T,mu))
    np.savetxt(outputfile+f'alldatapredicted49rnd_{lowestrnd}.txt',data)


def predicting_all(dictionary,model,outputfile,rnd_max,testing_data='allResults49.txt'):

    # Load data

    #training data
    input, output = model.load_data(rnd_max, False)
    input, output = model.input_columns_to_training(input, output)


    testing_data= 'testing_set.txt'
    file1 = testing_data
    eta_all = np.genfromtxt(file1,dtype=np.float64)[:,7:14]
    mu_all = np.genfromtxt(file1,dtype=np.float64)[:,22:]
    T_all= np.genfromtxt(file1,dtype=np.float64)[:,21:22]


    # Loop through each rnd value
    for rnd in range(rnd_max+1):
        model.load_trained_model(rnd)
        input, output = model.load_data_no_scale(rnd, False)
        pred = model.predict(input)
        eta = input[0]
        T = input[1]
        free = pred[0]
        mu = pred[1]
        grad = pred[2]
        eigen = np.zeros(mu.shape)
        eigenvector = np.zeros(grad.shape)
        for i in range(len(grad)):
            eigen[i,:], eigenvector[i,:,:] = LA.eig(grad[i,:,:])
        data = np.hstack((eta,mu,free,eigen,T))
        np.savetxt(outputfile+f'trainingdata_prediction_model_{rnd}.txt',data)


        pred = model.predict([eta_all,T_all])
        free_all= pred[0]
        mu_all = pred[1]
        grad_all = pred[2]
        eigen_all = np.zeros(mu_all.shape)
        eigenvector_all = np.zeros(grad_all.shape)
        for i in range(len(grad_all)):
            eigen_all[i,:], eigenvector_all[i,:,:] = LA.eig(grad_all[i,:,:])
        data = np.hstack((eta_all,mu_all,free_all,eigen_all,T_all))
        np.savetxt(outputfile+f'testingdata_prediction_model_{rnd}.txt',data)









def loss(dictionary,model,outputfile,rnd_max,testing_data='allResults49.txt'):


    # Load data

    #training data
    input, output = model.load_data(rnd_max, False)
    input, output = model.input_columns_to_training(input, output)


    testing_data= 'testing_set.txt'
    file1 = testing_data
    eta_all = np.genfromtxt(file1,dtype=np.float64)[:,7:14]
    mu_all = np.genfromtxt(file1,dtype=np.float64)[:,22:]
    T_all= np.genfromtxt(file1,dtype=np.float64)[:,21:22]
    data = np.hstack((eta_all,T_all,mu_all))
    i,o = model.array_to_column(data)
    i,o = model.scale_loaded_data(i,o)
    i,o = model.input_columns_to_training(i, o)



    # Initialize lists to store rnd values and corresponding losses
    rnd_values = []
    lastloss_values = []
    rndloss_values=[]
    allloss_values=[]
    # graphpoints_values=[]
    lowestloss = 10000
    lowestrnd=0

    # Loop through each rnd value
    for rnd in range(rnd_max+1):
        model.load_trained_model(rnd)
        # data = model.load_data(rnd_max)
        
        # Evaluate model and get loss
        print('evaluating')
        lastloss = model.model_evaluate(input, output)
        # print('lastlossmse',np.mean(((model.predict([input[1],input[3]]))[1]-output[1]) ** 2))
        # graphpointsloss = model.model_evaluate(g_input,g_output)

        allloss = model.model_evaluate(i,o)
        rndloss = model.loss(rnd,True)
        allpredict =model.predict([eta_all,T_all])
        print('alllossmse',np.mean((allpredict[1]*100-mu_all*100) ** 2))
        print('allpredict',allpredict[1])
        
        # Append rnd and loss values to lists
        rnd_values.append(rnd)
        lastloss_values.append(lastloss[2])
        rndloss_values.append(rndloss[2])
        allloss_values.append(allloss[2])
        # graphpoints_values.append(graphpointsloss[2])
        
        print('rnd', rnd)
        print('loss', lastloss)
        print('loss', rndloss)
        print('loss', allloss)
        if lastloss[2] < lowestloss:
            lowestloss=lastloss[2]
            lowestrnd = rnd

    print('lowest rnd', lowestrnd)
    model.load_trained_model(lowestrnd)
    input, output = model.load_data_no_scale(rnd, False)
    pred = model.predict(input)
    eta = input[0]
    T = input[1]
    free = pred[0]
    mu = pred[1]
    grad = pred[2]
    eigen = np.zeros(mu.shape)
    eigenvector = np.zeros(grad.shape)
    for i in range(len(grad)):
        eigen[i,:], eigenvector[i,:,:] = LA.eig(grad[i,:,:])
    data = np.hstack((eta,mu,free,eigen,T))
    np.savetxt(outputfile+f'trainingdata_prediction_model_{lowestrnd}.txt',data)


    pred = model.predict([eta_all,T_all])
    free_all= pred[0]
    mu_all = pred[1]
    grad_all = pred[2]
    eigen_all = np.zeros(mu_all.shape)
    eigenvector_all = np.zeros(grad_all.shape)
    for i in range(len(grad_all)):
        eigen_all[i,:], eigenvector_all[i,:,:] = LA.eig(grad_all[i,:,:])
    data = np.hstack((eta_all,mu_all,free_all,eigen_all,T_all))
    np.savetxt(outputfile+f'testingdata_prediction_model_{lowestrnd}.txt',data)


    # Plot rnd vs loss
    plt.figure(figsize=(10, 6))
    plt.plot(rnd_values, lastloss_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Rnd')
    plt.ylabel('Loss')
    plt.title('Loss vs. Rnd for Final Training Data Set')
    plt.grid(True)
    plt.savefig(outputfile+'mse_trainingdata.png')
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.plot(rnd_values, rndloss_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Rnd')
    plt.ylabel('Loss')
    plt.title('Loss vs. Rnd for Training Data of that rnd')
    plt.grid(True)
    plt.savefig(outputfile+'mse_rnddata.png')
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.plot(rnd_values, allloss_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Rnd')
    plt.ylabel('Loss')
    plt.title('Loss vs. Rnd for Testing Datas')
    plt.grid(True)
    plt.savefig(outputfile+'mse_testingdata.png')
    plt.clf()
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.plot(rnd_values, graphpoints_values, marker='o', linestyle='-', color='b')
    # plt.xlabel('Rnd')
    # plt.ylabel('Loss')
    # plt.title('Loss vs. Rnd for graph points data')
    # plt.grid(True)
    # plt.savefig(outputfile+'mse_graphdata.png')
    # plt.clf()
    # plt.show()

    np.savetxt(outputfile+'mse.txt',np.vstack((rnd_values,lastloss_values,rndloss_values,allloss_values)))



