import os
import sys, shutil
import numpy as np
import matplotlib.pyplot as plt
from active_learning.workflow.dictionary import Dictionary
from active_learning.model.idnn_model import IDNN_Model
from numpy import linalg as LA
from active_learning.workflow.make_graph import predict_and_save

def plotting_points(dictionary,model,outputFolder,rnd_max):
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
    plt.savefig(outputFolder+'data_sampled.png')
    plt.clf()

def predicting_best(dict,model,rnd,lowestrnd=None):
    [outputFolder,testing_data, dim] = dict.get_individual_keys('Main',['outputfolder','testing_set','derivative_dim'])

    if lowestrnd==None:
        mse = np.loadtxt(outputFolder+'mse.txt')
        last_loss = mse[1,:]
        lowestrnd  = np.argmin(last_loss)
    input, output = model.load_data_no_scale(rnd, False)
    eta = input[0]
    T = input[1]

    print('Rnd with lowest MSE', lowestrnd)
    model.load_trained_model(lowestrnd)
    predict_and_save(model,eta,T,outputFolder+f'LastRndData_rnd{lowestrnd}model.txt',f'LastRndData_rnd{lowestrnd}model')


    file1 = testing_data
    eta_all = np.genfromtxt(file1,dtype=np.float64)[:,dim:dim*2]
    mu_all = np.genfromtxt(file1,dtype=np.float64)[:,dim*3+1:]
    T_all= np.genfromtxt(file1,dtype=np.float64)[:,dim*3:dim*3+1]

    predict_and_save(model,eta_all,T_all,outputFolder+f'TestingData_rnd{lowestrnd}model.txt',f'TestingData_rnd{lowestrnd}model',mu_all)


def predicting_all(dict,model,rnd_max):
    
    
    [outputFolder,testing_data, dim] = dict.get_individual_keys('Main',['outputfolder','testing_set','derivative_dim'])


    #training data
    input, output = model.load_data(rnd_max, False)
    input, output = model.input_columns_to_training(input, output)


    file1 = testing_data
    eta_all = np.genfromtxt(file1,dtype=np.float64)[:,dim:dim*2]
    mu_all = np.genfromtxt(file1,dtype=np.float64)[:,dim*3+1:]
    T_all= np.genfromtxt(file1,dtype=np.float64)[:,dim*3:dim*3+1]


    # Loop through each rnd value
    for rnd in range(rnd_max+1):
        model.load_trained_model(rnd)
        input, output = model.load_data_no_scale(rnd, False)
        eta = input[0]
        T = input[1]
        predict_and_save(model,eta_all,T_all,outputFolder+f'TrainingData_rnd{rnd}model.txt',f'TrainingData_rnd{rnd}model')
        predict_and_save(model,eta_all,T_all,outputFolder+f'TestingData_rnd{rnd}model.txt',f'TestingData_rnd{rnd}model',mu_all)
        





def loss(dict,model,outputFolder,rnd_max,testing_data='allResults49.txt'):


    [outputFolder,testing_data, dim] = dict.get_individual_keys('Main',['outputfolder','testing_set','derivative_dim'])

    #training data
    input, output = model.load_data(rnd_max, False)
    input, output = model.input_columns_to_training(input, output)


    file1 = testing_data
    eta_all = np.genfromtxt(file1,dtype=np.float64)[:,dim:dim*2]
    mu_all = np.genfromtxt(file1,dtype=np.float64)[:,dim*3+1:]
    T_all= np.genfromtxt(file1,dtype=np.float64)[:,dim*3:dim*3+1]
    data = np.hstack((eta_all,T_all,mu_all))
    i,o = model.array_to_column(data)
    i,o = model.scale_loaded_data(i,o)
    i,o = model.input_columns_to_training(i, o)



    # Initialize lists to store rnd values and corresponding losses
    rnd_values = []
    lastloss_values = []
    rndloss_values=[]
    allloss_values=[]
    lowestloss = 10000
    lowestrnd=0

    # Loop through each rnd value
    for rnd in range(rnd_max+1):
        model.load_trained_model(rnd)
        # data = model.load_data(rnd_max)
        
        # Evaluate model and get loss
        print('Evaluating model for rnd',rnd)
        lastloss = model.model_evaluate(input, output)
        allloss = model.model_evaluate(i,o)
        rndloss = model.loss(rnd,True)
        allpredict =model.predict([eta_all,T_all])
        rnd_values.append(rnd)
        if isinstance(lastloss, list):
            lastloss = lastloss[-1]
            rndloss = rndloss[-1]
            allloss = allloss[-1]
        lastloss_values.append(lastloss)
        rndloss_values.append(rndloss)
        allloss_values.append(allloss)
        if lastloss < lowestloss:
            lowestloss=lastloss
            lowestrnd = rnd

    print('Round with lowest MSE: ', lowestrnd)
    np.savetxt(outputFolder+'mse.txt',np.vstack((rnd_values,lastloss_values,rndloss_values,allloss_values)))

    predicting_best(dict,model,rnd_max,lowestrnd)

    # Plot rnd vs loss
    plt.figure(figsize=(10, 6))
    plt.plot(rnd_values, lastloss_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Rnd')
    plt.ylabel('Loss')
    plt.title('Loss vs. Rnd for Final Training Data Set')
    plt.grid(True)
    plt.savefig(outputFolder+'mse_trainingdata.png')
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.plot(rnd_values, rndloss_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Rnd')
    plt.ylabel('Loss')
    plt.title('Loss vs. Rnd for Training Data of that rnd')
    plt.grid(True)
    plt.savefig(outputFolder+'mse_rnddata.png')
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.plot(rnd_values, allloss_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Rnd')
    plt.ylabel('Loss')
    plt.title('Loss vs. Rnd for Testing Datas')
    plt.grid(True)
    plt.savefig(outputFolder+'mse_testingdata.png')
    plt.clf()




