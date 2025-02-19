from active_learning.model.idnn import IDNN
from active_learning.model.idnn_model import IDNN_Model
from active_learning.workflow.dictionary import Dictionary
import numpy as np
# from CASM_wrapper import compileCASMOutput, loadCASMOutput
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from numpy import linalg as LA





def predict_and_save(model,eta,T,title,name,mu_real=None):
    print('Predicting points for ',name)
    pred = model.predict([eta,T])
    free = pred[0]
    mu = pred[1]
    grad = pred[2]
    eigen = np.zeros(mu.shape)
    eigenvector = np.zeros(grad.shape)
    for i in range(len(grad)):
        eigen[i,:], eigenvector[i,:,:] = LA.eig(grad[i,:,:])
    if mu_real is None:
        data = np.hstack((eta,mu,free,eigen,T))
    else:
        data = np.hstack((eta,mu,free,eigen,T))
    np.savetxt('{}.txt'.format(title),data)
    return data
    

def pred_training_points(rnd,model,dict):
    [outputFolder,temp,graph,dir_path, dim] = dict.get_individual_keys('Main',['outputfolder','temp','graph','dir_path','derivative_dim'])
    data = np.genfromtxt(outputFolder + 'data/data_sampled/CASMresults'+str(rnd)+'.txt',dtype=np.float64)
    eta = data[:,:dim]
    mu = data[:,-dim:]
    T = np.ones((np.shape(eta)[0],1))*temp
    data1= predict_and_save(model,eta,T,f'{outputFolder}/training/predictions/training_points_predicted_rnd{rnd}', 'training set')


def graph(rnd, model,dict):
    [outputFolder,temp,graph,dir_path, dim] = dict.get_individual_keys('Main',['outputfolder','temp','graph','dir_path','derivative_dim'])
    for g in graph:
        if g != 'none':
            data = dir_path + '/'+ g
            title = outputFolder+'graphs/{}_rnd{}'.format(g,rnd)
            
            eta = np.genfromtxt(data,dtype=np.float32)[:,:dim]
            T = np.ones((np.shape(eta)[0],1))*temp
            predict_and_save(model,eta,T,title, g)










