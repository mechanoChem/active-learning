
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
         self.N_sets, self.LR_range,self.Layers,self.Neurons,self.Dropout, self.Learning, self.validation_split] = self.dict.get_category_values('Neural Network')
        [self.outputFolder,self.input_alias, self.transform_path,
         self.output_alias,self.config_path, self.dim] = self.dict.get_individual_keys(['outputfolder',
        'input_alias','transforms_directory','output_alias','dir_path','Input_dim'])
        
        [self.WeightRecent,self.LR_decay,self.Min_lr,self.EarlyStopping,
          self.factor, self.Patience, self.lossterms, self.loss_weights] = self.dict.get_category_values('Training')
        # self.dim = np.size(self.input_alias)
        self.activation_list = []
        for i in range(len(self.hidden_units)):
            self.activation_list.append(random.choice(self.activation))

        for i in range(np.size(self.lossterms)):
            if self.lossterms[i] == 'None':
                self.lossterms[i] = None

        self.model = IDNN(self.dim,
                self.hidden_units,
                activation = self.activation_list,
                transforms=self.IDNN_transforms(),
                dropout=self.Dropout,
                unique_inputs=True,
                final_bias=True)
        self.opt = 'keras.optimizers.' + self.optimizer 
        self.model.compile(loss=self.lossterms,
                        loss_weights=self.loss_weights,
                        optimizer=eval(self.opt)(learning_rate=self.LR_range))
        self.learning_rate = np.power(10,(np.log10(self.LR_range[1]) - np.log10(self.LR_range[0]))*np.random.rand(1)[0] + np.log10(0.0001),dtype=np.float32)
    
    def predict(self,data):
        return self.model.predict(data)
    
    def loadOutput(self,rnd, dim, outputfolder, singleRnd=False):
        if singleRnd:
            return np.load(outputfolder + 'data/data_sampled/results{}.npy'.format(rnd),allow_pickle=True)
        else:
            return np.load(outputfolder + 'data/data_sampled/allResults{}.npy'.format(rnd),allow_pickle=True)
    

    def load_data(self,rnd,singleRnd=True):
        if singleRnd:
            input, input_non_derivative, output =  np.load(self.outputFolder + 'data/data_sampled/results{}.npy'.format(rnd),allow_pickle=True)
        else:
            input, input_non_derivative, output =  np.load(self.outputFolder + 'data/data_sampled/allResults{}.npy'.format(rnd),allow_pickle=True)
        
        j=0
        for i in range(np.size(self.input_alias)):
            _,_,derivative_dim,dimension,adjust = self.dict.get_category_values(self.input_alias[i])
            if derivative_dim:
                input[:,:,i] = (input[:,:,i]+adjust[0])*adjust[1]
            else:
                input_non_derivative[:,j] = (input_non_derivative[:,j]+adjust[0])*adjust[1]
                j+=1

        input = [input[0,:,:],input[1,:,:],input[2,:,:], input_non_derivative[0,:,:]]

        for i in range(np.size(self.output_alias)):
            derivative,dimensions,adjust = self.dict.get_category_values(self.output_alias[i])
            output[:][derivative] = (output[:][derivative]+adjust[0])*adjust[1]
        
        return input,output
        
    def loss(self,rnd):
        params = self.load_data(rnd,singleRnd=True) #Params[0] returns input, Params[1] returns output
        return self.model.evaluate(params[0],params[1])


    def load_model(self,rnd):
        model_name = 'model_' + rnd
        self.model = keras.models.load_model('idnn_1', custom_objects={'Transform': Transform(self.IDNN_transforms())})

    
    def new_model(self,hidden_units,lr):
        self.hidden_units = hidden_units
        self.learning_rate = lr
        self.model = IDNN(self.dim,
                self.hidden_units,
                activation = self.activation_list,
                transforms=self.IDNN_transforms(),
                dropout=self.Dropout,
                unique_inputs=True,
                final_bias=True)
        self.opt = 'keras.optimizers.' + self.optimizer 
        self.model.compile(loss=self.lossterms,
                        loss_weights=self.loss_weights,
                        optimizer=eval(self.opt)(learning_rate=self.LR_range))

    def train_rand_idnn(self,rnd,set_i):
        # rnd=0
        [layers_range,neurons_range] = self.dict.get_individual_keys(['layers_range','neurons_range'])
        input, output = self.load_data(rnd,singleRnd=True) 
        learning_rate = np.power(10,(np.log10(self.LR_range[1]) - np.log10(self.LR_range[0]))*np.random.rand(1)[0] + np.log10(0.0001),dtype=np.float32)
        n_layers = np.random.randint(layers_range[0],layers_range[1]+1)
        hidden_units = n_layers*[np.random.randint(neurons_range[0],neurons_range[1]+1)]

        rand_model = IDNN(self.dim,
                     hidden_units,
                     activation=self.activation_list,
                     transforms=self.IDNN_transforms(),
                     dropout=self.Dropout,
                     unique_inputs=True,
                     final_bias=True)
        self.opt = 'keras.optimizers.' + self.optimizer 
        # rand_model.compile(loss=self.lossterms,
        #                 loss_weights=self.loss_weights,
        #                 optimizer=eval(self.opt)(learning_rate=self.LR_range))

        model,valid_loss = self.surrogate_training(rnd,rand_model,input,output,set_i,learning_rate)

        model.save_weights(self.outputFolder+ 'training/model_{}_{}/model'.format(rnd,set_i))
       
        # print('Valid loss',valid_loss)

        return hidden_units, learning_rate, valid_loss
    
    def save_model(self,rnd):
        # print('saved model: ',self.outputFolder, 'training/model_',rnd)
        self.model.save_weights(self.outputFolder+ 'training/model_{}/model'.format(rnd))
        # ,save_format='tf'

    
    def train(self,rnd):
        input, output = self.load_data(rnd,singleRnd=True) 
        self.mode,_ = self.surrogate_training(rnd,self.model,input,output)
    

    def surrogate_training(self,rnd,model,input,output,set_i=None,learning_rate='default'):

        if learning_rate == 'default':
            learning_rate = self.learning_rate
            

        inds = np.arange(input[0].shape[1])
        np.random.shuffle(inds)
        input2 = []
        for i in input:
            input2.append(i[:,inds].T)

        input = input2

        output2 = []
        for i in output:
            output2.append(i[:,inds].T)

        output = output2

        if self.WeightRecent == 'Yes':
            # weight the most recent high error points as high as all the other points
            n_points = len(input)
            sample_weight = np.ones(n_points)
            if rnd > 0:
                sample_weight[-1000:] = max(1,(n_points-1000)/(2*1000))
            sample_weight = sample_weight[inds]


        # train
        lr_decay = self.LR_decay**rnd
        model.compile(loss=self.lossterms,
                        loss_weights=self.loss_weights,
                        optimizer=eval(self.opt)(learning_rate=learning_rate*lr_decay))
        if set_i==None:
            csv_logger = CSVLogger(self.outputFolder+'training/training_{}.txt'.format(rnd),append=True)
        else:
            csv_logger = CSVLogger(self.outputFolder+'training/training_{}_{}.txt'.format(rnd,set_i),append=True)

        reduceOnPlateau = ReduceLROnPlateau(factor=self.factor,patience=self.Patience,min_lr=self.Min_lr)
        callbackslist = [csv_logger, reduceOnPlateau]
        if EarlyStopping == 'Yes':
            earlyStopping = EarlyStopping(patience=self.Patience)
            callbackslist.append(earlyStopping)

        print('Training...')
        # for cols in input:
        #     print(np.shape(cols))
        # print('output')
        # for col in output:
        #     print(np.shape(col))


        # print('input',np.shape(input))
        # print('output',np.shape(output))
        if self.WeightRecent == 'Yes':
            history =model.fit(input,
                    output,
                    validation_split=self.validation_split,
                    epochs=self.Epochs,
                    batch_size=self.batch_size,
                    sample_weight=[sample_weight,sample_weight],
                    callbacks=callbackslist)
        else:
            history=model.fit(input,
                    output,
                    validation_split=self.validation_split,
                    epochs=self.Epochs,
                    batch_size=self.batch_size,
                    callbacks=callbackslist)
        valid_loss = history.history['val_loss'][-1]
        return model, valid_loss
    
    def IDNN_transforms(self):

        # sys.path.append(self.config_path)
        sys.path.append(self.config_path+self.transform_path)
        from TransformsModule import transforms 

        return transforms