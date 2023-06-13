
from active_learning.model.model import Model 
from active_learning.model.idnn import IDNN 
from tensorflow import keras 
from active_learning.model.transform_layer import Transform
import numpy as np
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from active_learning.model.data_generation_wrapper import submitCASM, compileCASMOutput, loadCASMOutput
import sys
import random


class IDNN_Model(Model):
    def __init__(self, dict):
        super().__init__()
        self.dict =dict

        [self.Epochs, self.batch_size,self.activation,self.hidden_units, self.optimizer, 
         self.N_sets, self.LR_range,self.Layers,self.Neurons,self.Dropout, self.Learning] = self.dict.get_category_values('Neural Network')
        [self.dim,self.Output_folder,self.derivative_dim,self.temp] = self.dict.get_individual_keys(['input_dim','outputfolder','derivative_dim','temperatures'])
        
        [self.WeightRecent,self.LR_decay,self.Min_lr,self.EarlyStopping,self.Patience] = self.dict.get_category_values('Training')
        self.Tmin = min(self.temp)
        self.Tmax = max(self.temp)
        self.activation_list = []
        for i in range(len(self.hidden_units)):
            self.activation_list.append(random.choice(self.activation))

        self.model = IDNN(self.dim,
                self.hidden_units,
                activation = self.activation_list,
                transforms=self.IDNN_transforms(),
                dropout=self.Dropout,
                unique_inputs=True,
                final_bias=True)
        self.opt = 'keras.optimizers.' + self.optimizer 
        self.model.compile(loss=['mse','mse',None],
                        loss_weights=[0.01,1,None],
                        optimizer=eval(self.opt)(learning_rate=self.LR_range))
    
    def predict(self,data):
        return self.model.predict(data)
    

    def load_data(self,rnd, singleRnd=True):
        return loadCASMOutput(rnd, self.dim, self.Output_folder, singleRnd=False)

        
    def loss(self,rnd):
        params = self.load_data(rnd,singleRnd=True) #Params[0] returns input, Params[1] returns output
        return self.model.evaluate(params[0],params[1])


    def load_model(self,rnd):
        model_name = 'model_' + rnd
        self.model = keras.models.load_model('idnn_1', custom_objects={'Transform': Transform(self.IDNN_transforms())})

    
    
    def save_model(self,rnd):
        self.model.save('model_{}'.format(rnd))

    
    def train(self,rnd):
        kappa,eta_train,mu_train,T_train = self.load_data(rnd,singleRnd=True) 
        learning_rate = np.power(10,(np.log10(self.LR_range[1]) - np.log10(self.LR_range[0]))*np.random.rand(1)[0] + np.log10(0.0001),dtype=np.float32)
        # print(params)
        # print('params shape ',np.shape(params))
        # eta_train = params[:self.derivative_dim]
        # T_train = params[self.derivative_dim]
        params = [eta_train,T_train]

        inds = np.arange(eta_train.shape[0])
        np.random.shuffle(inds)
        eta_train = eta_train[inds]
        mu_train = mu_train[inds]

        if self.WeightRecent == 'Yes':
            # weight the most recent high error points as high as all the other points
            n_points = len(eta_train)
            sample_weight = np.ones(n_points)
            if rnd > 0:
                sample_weight[-1000:] = max(1,(n_points-1000)/(2*1000))
            sample_weight = sample_weight[inds]

        # create energy dataset (zero energy at origin)
        eta_train0 = np.zeros(eta_train.shape)
        g_train0 = np.zeros((eta_train.shape[0],1))
        T_train0 = np.zeros(T_train.shape)

        #Adjust mu, T from -1 to 1 
        T_train = (T_train - ((self.Tmax - self.Tmin)/2))/(self.Tmax - ((self.Tmax - self.Tmin)/2))
        #for i in range(self.Dim):
        #    eta_train[i] 

        # train
        lr_decay = self.LR_decay**rnd
        self.model.compile(loss=['mse','mse',None],
                        loss_weights=[0.01,1,None],
                        optimizer=eval(self.opt)(lr=learning_rate*lr_decay))
        csv_logger = CSVLogger(self.Output_folder+'data/training/training_{}.txt'.format(rnd),append=True)
        reduceOnPlateau = ReduceLROnPlateau(factor=0.5,patience=100,min_lr=self.Min_lr)
        callbackslist = [csv_logger, reduceOnPlateau]
        if EarlyStopping == 'Yes':
            earlyStopping = EarlyStopping(patience=150)
            callbackslist.append(earlyStopping)

        print('Training...')



        if self.WeightRecent == 'Yes':
            history =self.model.fit([eta_train0,eta_train,0*eta_train, T_train],
                    [g_train0,100*mu_train,0*mu_train],
                    validation_split=0.25,
                    epochs=self.Epochs,
                    batch_size=self.batch_size,
                    sample_weight=[sample_weight,sample_weight],
                    callbacks=callbackslist)
        else:
            history=self.model.fit([eta_train0,eta_train,0*eta_train, T_train],
                    [g_train0,100*mu_train,0*mu_train],
                    validation_split=0.25,
                    epochs=self.Epochs,
                    batch_size=self.batch_size,
                    callbacks=callbackslist)
        return self.model
    
    def IDNN_transforms(self):

        # sys.path.append(self.config_path)
        from tests.LCO.TransformsModule import transforms 

        return transforms