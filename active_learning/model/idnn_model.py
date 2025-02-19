
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
import tensorflow as tf
import random



class IDNN_Model():
    def __init__(self, dict):
        super().__init__()
        self.dict =dict


        [self.layers,self.neurons,self.activation,self.dropout,self.transform_path,
         self.lossterms,self.loss_weights,self.optimizer,self.learning,self.lr_decay,
         self.factor, self.patience,self.min_lr,self.EarlyStopping,self.epochs,
         self.batch_size,self.WeightRecent,self.validation_split,
         self.hyperparameter,self.train_new_idnn] = self.dict.get_category_values('IDNN')
         
         
        with open(self.transform_path, "r") as f:
            self.transform = json.load(f)
        


        [self.input_alias,self.output_alias,self.config_path,self.outputFolder,self.dim] = self.dict.get_individual_keys('Main',['input_alias','output_alias','dir_path','outputfolder','input_dim'])
        self.hidden_units = self.layers*[self.neurons]
        self.activation_list = []
        for i in range(len(self.hidden_units)):
            self.activation_list.append(random.choice(self.activation))

        for i in range(np.size(self.lossterms)):
            if self.lossterms[i] == 'None':
                self.lossterms[i] = None
    

        [self.QBC]  = self.dict.get_individual_keys('Exploit_Parameters',['qbc'])
    

        self.model = IDNN(self.dim-1,
                self.hidden_units,
                activation = self.activation_list,
                transforms=self.IDNN_transforms(),
                dropout=self.dropout,
                unique_inputs=True,
                final_bias=True)
        self.opt = 'keras.optimizers.' + self.optimizer 


        self.model.compile(loss=self.lossterms,
                        loss_weights=self.loss_weights,
                        optimizer=eval(self.opt)(learning_rate=self.learning))

    def predict(self,data):
        data= self.scale_loaded_data(data)
        data = [data[0],data[0],data[0],data[1]]
        output = self.model.predict(data)
        return self.scale_output_back(output)
    

    def array_to_column(self, data):
        column_list_input = []
        column_list_output = []
        position = 0
        for inputs in self.input_alias:
            [dim] =self.dict.get_individual_keys(inputs,['dimensions'])
            column_list_input.append(data[:,position:dim+position])
            position+=dim

        for output in self.output_alias:
            [dim] =self.dict.get_individual_keys(output,['dimensions'])
            column_list_output.append(data[:,position:dim+position])
            position+=dim
        return column_list_input,column_list_output
            

    def scale_loaded_data(self,inputs, output=None):
        #Switch to columns
        # input, output = self.array_to_column(input,output) 

        for i in range(np.size(self.input_alias)):
            _,_,derivative_dim,dimension,adjust = self.dict.get_category_values(self.input_alias[i])
            temp = (inputs[i]+adjust[0])*adjust[1] 
            inputs[i] = temp
        if output == None:
            return inputs
        else:
            for i in range(np.size(self.output_alias)):
                derivative,dimensions,adjust = self.dict.get_category_values(self.output_alias[i])
                output[:][i] = (output[:][i]+adjust[0])*adjust[1]
                temp = (output[i]+adjust[0])*adjust[1] 
                output[i] = temp
            return inputs,output

    def scale_output_back(self,output):
        derivative,dimensions,adjust = self.dict.get_category_values(self.output_alias[0])
        for i in range(3):
            temp = (output[i]/adjust[1])-adjust[0]
            output[i] = temp
        return output

    

    def load_data(self,rnd,singleRnd=True):
        if singleRnd:
            data =  np.load(self.outputFolder + 'data/data_sampled/results{}.npy'.format(rnd),allow_pickle=True)
        else:
            data =  np.load(self.outputFolder + 'data/data_sampled/allResults{}.npy'.format(rnd),allow_pickle=True)

        input, output = self.array_to_column(data)

        return self.scale_loaded_data(input, output)
        
    def loss(self,rnd,singleRnd=True):
        inputs,output = self.load_data(rnd,singleRnd) #Params[0] returns input, Params[1] returns output
        inputs,output = self.input_columns_to_training(inputs,output)
        return self.model_evaluate(inputs,output)

        
    def model_evaluate(self,input,output):
        return self.model.evaluate(input,output)

    def set_params(self,rnd):
        with open(self.outputFolder + 'training/model_{}/params.json'.format(rnd-1)) as json_file:
            params = json.load(json_file)
        [self.layers,self.neurons,self.activation_list,self.dropout,self.optimizer,self.learning,self.lr_decay,self.factor,self.patience,self.min_lr,self.epochs,self.batch_size] = params
        self.hidden_units = self.layers*[self.neurons]
        
        self.model = IDNN(self.dim,
                self.hidden_units,
                activation = self.activation_list,
                transforms=self.IDNN_transforms(),
                dropout=self.dropout,
                unique_inputs=True,
                final_bias=True)
        self.opt = 'keras.optimizers.RMSprop'
        self.model.compile(loss=self.lossterms,
                        loss_weights=self.loss_weights,
                        optimizer=eval(self.opt)(learning_rate=self.learning))


    def load_trained_model(self,rnd):
        with open(self.outputFolder + 'training/model_{}/params.json'.format(rnd)) as json_file:
            params = json.load(json_file)

        [self.layers,self.neurons,self.activation_list,self.dropout,self.optimizer,self.learning,self.lr_decay,self.factor,self.patience,self.min_lr,self.epochs,self.batch_size] = params
        self.hidden_units = self.layers*[self.neurons]
        
        loaded_model = IDNN(self.dim,
                self.hidden_units,
                activation = self.activation_list,
                transforms=self.IDNN_transforms(),
                dropout=self.dropout,
                unique_inputs=True,
                final_bias=True)
        self.opt = 'keras.optimizers.RMSprop'
        # loaded_model.load_weights(self.outputFolder+ 'training/model_{}/model.weights.h5'.format(rnd))
        loaded_model.compile(loss=self.lossterms,
                        loss_weights=self.loss_weights,
                        optimizer=eval(self.opt)(learning_rate=self.learning))

        etas = np.zeros((2,self.dim))
        T = np.zeros((2,1))
        loaded_model.fit([etas,etas,etas,T],[etas,etas,etas],epochs=2,verbose=0)
        # print(self.outputFolder+ 'training/model_{}/model.weights.h5'.format(rnd))
        loaded_model.load_weights(self.outputFolder+ 'training/model_{}/model.weights.h5'.format(rnd))        
        self.model = loaded_model
        print('Loaded trained model rnd',rnd)
        return loaded_model

    def parameter_int(self,keys):
        list = []
        for key in keys:
            [value] = self.dict.get_individual_keys('IDNN_Hyperparameter',[key])
            if value == None:
                list += [eval('self.'+key)]
            else:
                list += [np.random.randint(value[0],value[1]) ]
        return list
    
    def parameter_float(self,keys):
        list = []
        for key in keys:
            [value] = self.dict.get_individual_keys('IDNN_Hyperparameter',[key])
            if value == None:
                list += [eval('self.'+key)]
            else:
                list += [np.random.uniform(value[0],value[1]) ]
        return list
    

            

    def train_rand_idnn(self,rnd,set_i):
        [layers,neurons,epochs,batch_size] = self.parameter_int(['layers','neurons','epochs','batch_size'])
        [dropout,learning,lr_decay,factor,patience,min_lr] = self.parameter_float(['dropout','learning','lr_decay','factor','patience','min_lr'])

        inputs, output = self.load_data(rnd,singleRnd=False) 
        hidden_units = layers*[neurons]
        [activation,optimizer] = self.dict.get_individual_keys('IDNN_Hyperparameter',['activation','optimizer'])
        if activation == None:
            activation_list = self.activation_list
        else:
            activation_list = random.choices(activation,k=layers)
        if optimizer == None:
            optimizer = self.optimizer
        else:
            optimizer = random.choice(optimizer)

        rand_model = IDNN(self.dim,
                     hidden_units,
                     activation=activation_list,
                     transforms=self.IDNN_transforms(),
                     dropout=dropout,
                     unique_inputs=True,
                     final_bias=True)
        opt = 'keras.optimizers.' + optimizer 
        rand_model.compile(loss=self.lossterms,
                        loss_weights=self.loss_weights,
                        optimizer=eval(opt)(learning_rate=learning*lr_decay))

        rand_model,valid_loss = self.surrogate_training(rnd,rand_model,inputs,output,set_i,learning)

        params = [layers,neurons,activation_list,dropout,optimizer,learning,lr_decay,factor,patience,min_lr,epochs,batch_size]
        os.makedirs(self.outputFolder+ 'training/model_{}_{}'.format(rnd,set_i))
        rand_model.save_weights(self.outputFolder+ 'training/model_{}_{}/model.weights.h5'.format(rnd,set_i))
        jsonparams = json.dumps(params)
        with open(self.outputFolder + 'training/model_{}_{}/params.json'.format(rnd,set_i), "w") as outfile:
            outfile.write(jsonparams)


        if self.QBC:
            data = np.loadtxt(self.outputFolder+'data/data_recommended/global_prediction_points_rnd'+str(rnd)+'.txt')
            prediction  = self.predict([data[:,0:self.dim-1],data[:,self.dim-1:]])
            pred = np.hstack((prediction[0],prediction[1]))
            np.savetxt(self.outputFolder + 'training/predictions/prediction_{}_{}.json'.format(rnd,set_i),pred) 


        
            

        return valid_loss,params
    
    def save_model(self,rnd):
        os.makedirs(self.outputFolder+ 'training/model_{}'.format(rnd))
        self.model.save_weights(self.outputFolder+ 'training/model_{}/model.weights.h5'.format(rnd))


        params = [self.layers,self.neurons,self.activation_list,self.dropout,self.optimizer,self.learning,self.lr_decay,self.factor,self.patience,self.min_lr,self.epochs,self.batch_size]
        jsonparams = json.dumps(params)
        with open(self.outputFolder + 'training/model_{}/params.json'.format(rnd), "w") as outfile:
            outfile.write(jsonparams)

    
    def train(self,rnd):
        inputs, output = self.load_data(rnd,singleRnd=False) 
        if self.train_new_idnn and rnd>0:
            self.set_params(rnd)


        self.model,valid_loss = self.surrogate_training(rnd,self.model,inputs,output)
        return valid_loss
    

    def input_columns_to_training(self,inputs,output=None, unique_inputs=True):
        input_new = []
        output_new = []
        input_non_derivative = []
        j=0
        k=0
        for i in range(np.size(self.input_alias)):
            _,_,derivative_dim,dimension,adjust = self.dict.get_category_values(self.input_alias[i])
            if derivative_dim:
                if j==0:
                    input_new = inputs[:][i]
                else:
                    input_new = np.hstack((input_new,inputs[:][i]))
                j+=1
            else:
                if k==0:
                    input_non_derivative =inputs[:][i]
                else:
                    input_non_derivative = np.hstack((input_non_derivative,inputs[:][i]))
                k+=1
        input_zeros = np.zeros(np.shape(input_new))
        if unique_inputs:
            inputs = [input_zeros,input_new,input_new,input_non_derivative]
        else:
            inputs = [input_new,input_non_derivative]

        if output == None:
            return inputs

        else:
            for i in range(np.size(self.output_alias)):
                if i==0:
                    output_new = output[:][i]
                else:
                    output_new = np.hstack((output_new,output[:][i]))

            # return inputs, [input_zeros,output_new,output_new*0 ]
            return inputs, [np.zeros((np.shape(output_new)[0],1)),output_new,output_new*0 ]


    def surrogate_training(self,rnd,model,inputs,output,set_i=None,learning_rate='default'):


        if learning_rate == 'default':
            learning_rate = self.learning

        inputs,output = self.input_columns_to_training(inputs,output)
        inds = np.arange(inputs[0].shape[1])

        if self.WeightRecent:
            # weight the most recent high error points as high as all the other points
            n_points = inputs[0].shape[0]#len(inputs[0])
            sample_weight = np.ones(n_points)
            i,o = self.load_data(rnd,singleRnd=True)
            recentpoints = i[0].shape[0]
            if rnd > 0:
                sample_weight[-recentpoints:] = max(1,(n_points-recentpoints)/(recentpoints))



        # trains
        lr_decay = self.lr_decay**rnd
        model.compile(loss=self.lossterms,
                        loss_weights=self.loss_weights,
                        optimizer=eval(self.opt)(learning_rate=learning_rate*lr_decay))
        if set_i==None:
            csv_logger = CSVLogger(self.outputFolder+'training/trainings/training_{}.txt'.format(rnd),append=True)
        else:
            csv_logger = CSVLogger(self.outputFolder+'training/trainings/training_{}_{}.txt'.format(rnd,set_i),append=True)

        reduceOnPlateau = ReduceLROnPlateau(factor=self.factor,patience=self.patience,min_lr=self.min_lr)
        selective_logger= SelectiveProgbarLogger(verbose=1, epoch_interval=50)
        callbackslist = [csv_logger, reduceOnPlateau,selective_logger]
        if EarlyStopping:
            earlyStopping = EarlyStopping(patience=self.patience*2)
            callbackslist.append(earlyStopping)


        
        if self.WeightRecent:
            history =model.fit(inputs,
                    output,
                    validation_split=self.validation_split,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    sample_weight=[sample_weight,sample_weight,sample_weight],
                    callbacks=callbackslist,
                    verbose=0)
        
        else:
            history=model.fit(inputs,
                    output,
                    validation_split=self.validation_split,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=callbackslist,
                    verbose=0)
        valid_loss = history.history['val_loss'][-1]
        return model, valid_loss
    
    def load_data_no_scale(self,rnd,singleRnd=True):
        if singleRnd:
            data =  np.load(self.outputFolder + 'data/data_sampled/results{}.npy'.format(rnd),allow_pickle=True)
        else:
            data =  np.load(self.outputFolder + 'data/data_sampled/allResults{}.npy'.format(rnd),allow_pickle=True)
        
        input, output = self.array_to_column(data)

        return input,output

    def IDNN_transforms(self):

        def transforms(x):
            return [eval(expr, {"x": x, "np": np}) for _, expr in self.transform]

        return transforms
    
class SelectiveProgbarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, verbose, epoch_interval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_verbose = verbose
        self.epoch_interval = epoch_interval
    
    def on_epoch_begin(self, epoch, *args, **kwargs):
        self.verbose = (
            0 
                if epoch % self.epoch_interval != 0 
                else self.default_verbose
        )
        super().on_epoch_begin(epoch, *args, **kwargs)