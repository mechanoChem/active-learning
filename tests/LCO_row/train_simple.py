# from tensorflow import keras
import sys, os

import numpy as np
from numpy import linalg as LA
# from mechanoChemML.src.idnn import IDNN
# from mechanoChemML.workflows.active_learning.hitandrun import billiardwalk
#from mechanoChemML.src.transform_layer import Transform
import json
import matplotlib.pyplot as plt
# import plotly.express as px
# import pandas as pd
import sys
directory = os.path.abspath('../../../active-learning/')
sys.path.insert(0,directory)

from active_learning.model.idnn import IDNN
from tensorflow import keras 
from active_learning.model.transform_layer import Transform
import numpy as np
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from active_learning.model.data_generation_wrapper import submitCASM, compileCASMOutput, loadCASMOutput
import sys
import random
from tensorflow.keras.models import load_model
import json,os


def IDNN_transforms():

    # sys.path.append(self.config_path)
    # sys.path.append(self.config_path+self.transform_path)
    # from TransformsModule import transforms 
    def transforms(x):    
        h0 = x[:,0]
        h1 = 1./3.*(x[:,1]**2 + x[:,2]**2 + x[:,3]**2)
        h2 = 1./3.*(x[:,1]**4 + x[:,2]**4 + x[:,3]**4)
        h3 = 1./3.*((x[:,1]**2)*(x[:,2]**2 + x[:,3]**2) + (x[:,2]**2*x[:,3]**2))
        h4 = 1./3.*(x[:,1]**6 + x[:,2]**6 + x[:,3]**6)
        h5 = 1./6.*((x[:,1]**4)*( x[:,2]**2 + x[:,3]**2) +(x[:,2]**4)*( x[:,1]**2 + x[:,3]**2)+(x[:,3]**4)*( x[:,2]**2 + x[:,1]**2)  )            
        h6 = (x[:,1]**2)*(x[:,2]**2)*(x[:,3]**2)
        
        return [h0,h1,h2,h3,h4,h5,h6]

    return transforms

file1 = '../../Output/Output8/CASMallResults15.txt'
eta = np.genfromtxt(file1,dtype=np.float32)[:,4:8]
mu = np.genfromtxt(file1,dtype=np.float32)[:,13:17]*100
T= np.genfromtxt(file1,dtype=np.float32)[:,12:13]*0
# T = np.reshape(T, (size()))

for i in range(5,6):

    dim=4
    layers = 3
    neurons = 150
    hidden_units = [neurons]*(layers)
    lr = .0001
    dropout=0.01
    learning=lr
    activation = 'tanh'
    activation_list = []
    lr_decay = 1*i
    for k in range(len(hidden_units)):
        activation_list.append(activation)

    # # print(activation_list)


    

    model =IDNN(2,
            [neurons]*(layers),
            activation = activation_list,
            dropout=dropout,
            unique_inputs=True,
            final_bias=True)
    
    print("hidden_units model 1",hidden_units)
    model = IDNN(dim,
        hidden_units,
        activation = activation_list,
        transforms=IDNN_transforms(),
        dropout=dropout,
        unique_inputs=True,
        final_bias=True)
    
    optimizer = 'RMSprop'
    opt = 'keras.optimizers.' + optimizer 
    lossterms=['mse','mse',None]
    loss_weights= [0.1,10,0]
    model.compile(loss=lossterms,
                        loss_weights=loss_weights,
                        optimizer=eval(opt)(clipvalue=5.0,learning_rate=learning))
    

    csv_logger = CSVLogger('../../Output/Output8/training_{}.txt'.format(i),append=True)
    eta_train0 = np.zeros(eta.shape)
    g_train0 = np.zeros((eta.shape[0],1))
    T = np.zeros((eta.shape[0],1))

    # eta[:,0] = (eta[:,0] - .5)*2
    # eta[:,1] = (eta[:,1])*2

    inputs = [eta_train0,eta,eta,T]
    outputs= [g_train0,mu*100,0*mu]

#     # print('inputs',inputs)

    epochs=100
    batch_size=10
    
    factor=0.5
    patience = 150
    min_lr = 1.e-6
    

    reduceOnPlateau = ReduceLROnPlateau(factor=factor,patience=patience,min_lr=min_lr)
    callbackslist = [csv_logger]

    history = model.fit(inputs,
                outputs,
                validation_split=0.25,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbackslist)

    valid_loss = history.history['val_loss'][-1]
    
    os.mkdir('../../Output/Output8/model_{}'.format(i))
    
    np.savetxt('../../Output/Output8/params_{}'.format(i),[layers,neurons,lr,valid_loss])

    params = [layers,neurons,activation_list,dropout,'RMSprop',learning,lr_decay,factor,patience,min_lr,epochs,batch_size]
    model.save_weights('../../Output/Output8/model_{}/model.weights.h5'.format(i))
    
    jsonparams = json.dumps(params)
    with open('../../Output/Output8/model_{}/params.json'.format(i), "w") as outfile:
        outfile.write(jsonparams)

    # with open('../../Output/Output8/model_{}/params.json'.format(i)) as json_file:
    #     params = json.load(json_file)

    # [layers,neurons,activation_list,dropout,optimizer,learning,lr_decay,factor,patience,min_lr,epochs,batch_size] = params
    # opt = 'keras.optimizers.' + optimizer 
    # hidden_units = [neurons]*(layers)
    # print("hidden_units model 2",hidden_units)
    # dim=4
    # lossterms=['mse','mse',None]
    # loss_weights= [0.1,10,0]

    # model2 = IDNN(dim,
    #     hidden_units,
    #     activation = activation_list,
    #     transforms=IDNN_transforms(),
    #     dropout=dropout,
    #     unique_inputs=True,
    #     final_bias=True)
    
    # model2.compile(loss=lossterms,
    #                 loss_weights=loss_weights,
    #                 optimizer=eval(opt)(learning_rate=learning))


    # history = model2.fit(inputs,
    #             outputs,
    #             validation_split=0.25,
    #             epochs=2,
    #             batch_size=batch_size)

    # model2.build(input_shape=(None, neurons,neurons,neurons,neurons))
    
    # model2.load_weights('../../Output/Output8/model_{}/model.weights.h5'.format(i))
    

    pred = model.predict([eta,eta,eta,T])
    g = pred[0]/100
    mu = pred[1]/100
#     eta[:,0] = eta[:,0]/20+.5
    # eta[:,1] = eta[:,1]/4+.25
    np.savetxt('../../Output/Output8/hp_prediction_{}.txt'.format(i),np.hstack((eta,mu,g)))
