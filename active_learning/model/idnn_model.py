
from active_learning.model.model import Model 
from active_learning.model.idnn import IDNN 
from tensorflow import keras 
from active_learning.model.transform_layer import Transform
import numpy as np
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from active_learning.model.data_generation_wrapper import submitCASM, compileCASMOutput, loadCASMOutput
import sys


class IDNN_Model(Model):
    def __init__(self, dict):
        super().__init__()
        self.dict =dict

        [self.epochs, self.batch_size,self.activation,self.hidden_units, self.optimizer, 
         self.N_sets, self.LearningRate,self.Layers,self.Neurons,self.Dropout, self.Learning] = self.dict.get_category_values('Neural Network')
        [self.dim] = self.dict.get_individual_keys(['input_dim'])
        self.model = IDNN(self.dim,
                self.hidden_units,
                activation = self.activation,
                transforms=self.IDNN_transforms(),
                dropout=self.Dropout,
                unique_inputs=True,
                final_bias=True)
        self.opt = 'keras.optimizers.' + self.optimizer 
        self.model.compile(loss=['mse','mse',None],
                        loss_weights=[0.01,1,None],
                        optimizer=eval(self.opt)(learning_rate=self.LearningRate))
    
    def predict(self,data):
        output = self.model.predict(input)
    

    def load_data(self,rnd, singleRnd=True):
        loadCASMOutput(rnd, self.dim, singleRnd=False)

        
    def loss(self,rnd):
        params = self.load_data(rnd,singleRnd=True) #Params[0] returns input, Params[1] returns output
        return self.model.evaluate(params[0],params[1])


    def load_model(self,rnd):
        model_name = 'model_' + rnd
        self.model = keras.models.load_model('idnn_1', custom_objects={'Transform': Transform(self.IDNN_transforms())})

    
    
    def save_model(self,rnd):
        self.model.save('model_{}{}'.format(rnd))

    
    def train(self,rnd):
        params = self.load_data(rnd,singleRnd=True) 

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
        print('SHAPE: ',eta_train.shape)
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
                        optimizer=eval(self.opt)(lr=self.lr*lr_decay))
        csv_logger = CSVLogger('training/training_{}{}.txt'.format(rnd,set_i),append=True)
        reduceOnPlateau = ReduceLROnPlateau(factor=0.5,patience=100,min_lr=self.Min_lr)
        callbackslist = [csv_logger, reduceOnPlateau]
        if EarlyStopping == 'Yes':
            earlyStopping = EarlyStopping(patience=150)
            callbackslist.append(earlyStopping)

        print('Training...')


        if self.WeightRecent == 'Yes':
            history =self.model.fit(params[0],
                    params[1],
                    validation_split=0.25,
                    epochs=self.Epochs,
                    batch_size=self.Batch_size,
                    sample_weight=[sample_weight,sample_weight],
                    callbacks=callbackslist)
        else:
            history=self.model.fit(params[0],
                    params[1],
                    validation_split=0.25,
                    epochs=self.Epochs,
                    batch_size=self.Batch_size,
                    callbacks=callbackslist)
    
    def IDNN_transforms(self):

        # sys.path.append(self.config_path)
        from tests.LCO.TransformsModule import transforms 

        return transforms