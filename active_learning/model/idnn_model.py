
from active_learning.model.model import Model 
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


class IDNN_Model(Model):
    def __init__(self, dict):
        super().__init__()
        self.dict =dict

        [self.layers,self.neurons,self.activation,self.dropout,self.transform_path,
         self.lossterms,self.loss_weights,self.optimizer,self.learning,self.lr_decay,
         self.factor, self.patience,self.min_lr,self.EarlyStopping,self.epochs,
         self.batch_size,self.WeightRecent,self.validation_split,
         self.hyperparameter] = self.dict.get_category_values('IDNN')


        [self.input_alias,self.output_alias,self.config_path,self.outputFolder] = self.dict.get_individual_keys('Main',['input_alias','output_alias','dir_path','outputfolder'])
        self.dim = np.size(self.input_alias)
        self.hidden_units = self.layers*[self.neurons]
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
                dropout=self.dropout,
                unique_inputs=True,
                final_bias=True)
        self.opt = 'keras.optimizers.' + self.optimizer 
        self.model.compile(loss=self.lossterms,
                        loss_weights=self.loss_weights,
                        optimizer=eval(self.opt)(learning_rate=self.learning))
        # self.learning_rate = np.power(10,(np.log10(self.LR_range[1]) - np.log10(self.LR_range[0]))*np.random.rand(1)[0] + np.log10(0.0001),dtype=np.float32)
    
    def predict(self,data):
        data= self.scale_loaded_data(data)
        data = self.input_columns_to_training(data)
        print('predict')
        print('inputs',data)
        # print('outputs',output)
        output = self.model.predict(data)
        return self.scale_output_back(output)
    
    # def adjust_scaling(self,input,value):
    #     #input non derivative - temperature
    #     #input regular - eta's
    #     #output - free/mu

    #      [adjust]= self.dict.get_individual_keys(value,['adjust'])
    #      input = (input+adjust[0])*adjust[1]
    #      return input

    def array_to_column(self, data):
        #self.input_alias x,eta,T
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

        print('array to column')
        print(column_list_input)

        return column_list_input,column_list_output
            
        #dim should be [1,6,1]

        # model_order = self.dict.get_individual_keys('Ordering',['model_order'])


    def scale_loaded_data(self,inputs, output=None):
        #Switch to columns
        # input, output = self.array_to_column(input,output) 

        for i in range(np.size(self.input_alias)):
            # print('scaling ',self.input_alias[i])
            _,_,derivative_dim,dimension,adjust = self.dict.get_category_values(self.input_alias[i])
            # print(inputs[i] )
            temp = (inputs[i]+adjust[0])*adjust[1] 
            inputs[i] = temp
            # print(inputs[i])
        # exit()
        if output == None:
            return inputs
        else:
            for i in range(np.size(self.output_alias)):
                derivative,dimensions,adjust = self.dict.get_category_values(self.output_alias[i])
                output[:][i] = (output[:][i]+adjust[0])*adjust[1]
            return inputs,output

    def scale_output_back(self,output):
        for i in range(np.size(self.output_alias)):
            derivative,dimensions,adjust = self.dict.get_category_values(self.output_alias[i])
            output[:][i] = (output[:][i]/adjust[1])-adjust[0]
        return output

    

    def load_data(self,rnd,singleRnd=True):
        if singleRnd:
            data =  np.load(self.outputFolder + 'data/data_sampled/results{}.npy'.format(rnd),allow_pickle=True)
        else:
            data =  np.load(self.outputFolder + 'data/data_sampled/allResults{}.npy'.format(rnd),allow_pickle=True)
        
        input, output = self.array_to_column(data)

        return self.scale_loaded_data(input, output)
        
    def loss(self,rnd):
        inputs,output = self.load_data(rnd,singleRnd=True) #Params[0] returns input, Params[1] returns output
        inputs,output = self.input_columns_to_training(inputs,output)
        print('loss method')
        print('inputs',inputs)
        print('outputs',output)
        return self.model.evaluate(inputs,output)
        # input2 = []
        # for i in input:
        #     input2.append(i.T)

        # input = input2

        # output2 = []
        # for i in output:
        #     output2.append(i.T)

        # output = output2
        


    def load_model(self,rnd):
        # readparams = open(self.outputFolder + 'training/model_{}/params'.format(rnd))
        with open(self.outputFolder + 'training/model_{}/params.json'.format(rnd)) as json_file:
            params = json.load(json_file)
        # print('params:',params)
        [self.layers,self.neurons,self.activation_list,self.dropout,self.optimizer,self.learning,self.lr_decay,self.factor,self.patience,self.min_lr,self.epochs,self.batch_size] = params
        self.hidden_units = self.layers*[self.neurons]
        load_model = IDNN(self.dim,
                self.hidden_units,
                activation = self.activation_list,
                transforms=self.IDNN_transforms(),
                dropout=self.dropout,
                unique_inputs=True,
                final_bias=True)
        self.opt = 'keras.optimizers.' + self.optimizer 
        load_model.compile(loss=self.lossterms,
                        loss_weights=self.loss_weights,
                        optimizer=eval(self.opt)(learning_rate=self.learning))
        
        load_model.load_weights(self.outputFolder+ 'training/model_{}/model'.format(rnd)).expect_partial()
        self.model = load_model

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

        inputs, output = self.load_data(rnd,singleRnd=True) 
        # if learning != None:
        #     learning_rate = np.power(10,(np.log10(learning[1]) - np.log10(learning[0]))*np.random.rand(1)[0] + np.log10(0.0001),dtype=np.float32)
        # else:
        #     learning = self.learning
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

        # print('optimizer',optimizer)

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


        # inds = np.arange(input[0].shape[1])
        # np.random.shuffle(inds)
        # input2 = []
        # for i in input:
        #     input2.append(i[:,inds].T)

        # input = input2

        # output2 = []
        # for i in output:
        #     output2.append(i[:,inds].T)

        # output = output2

        # if self.WeightRecent == 'Yes':
        #     # weight the most recent high error points as high as all the other points
        #     n_points = len(input)
        #     sample_weight = np.ones(n_points)
        #     if rnd > 0:
        #         sample_weight[-1000:] = max(1,(n_points-1000)/(2*1000))
        #     sample_weight = sample_weight[inds]


        # # train
        # lr_decay = self.lr_decay**rnd
        # csv_logger = CSVLogger(self.outputFolder+'training/training_{}_{}.txt'.format(rnd,set_i),append=True)

        # reduceOnPlateau = ReduceLROnPlateau(factor=factor,patience=patience,min_lr=min_lr)
        # callbackslist = [csv_logger, reduceOnPlateau]
        # if EarlyStopping == 'Yes':
        #     earlyStopping = EarlyStopping(patience=patience)
        #     callbackslist.append(earlyStopping)

        # print('Training...')
        # # print(input)
        # # print(output)
        # if self.WeightRecent == 'Yes':
        #     history =rand_model.fit(input,
        #             output,
        #             validation_split=self.validation_split,
        #             epochs=epochs,
        #             batch_size=batch_size,
        #             sample_weight=[sample_weight,sample_weight],
        #             callbacks=callbackslist)
        # else:
        #     history=rand_model.fit(input,
        #             output,
        #             validation_split=self.validation_split,
        #             epochs=epochs,
        #             batch_size=batch_size,
        #             callbacks=callbackslist)
        # valid_loss = history.history['val_loss'][-1]
    
        params = [layers,neurons,activation_list,dropout,optimizer,learning,lr_decay,factor,patience,min_lr,epochs,batch_size]
        os.makedirs(self.outputFolder+ 'training/model_{}_{}'.format(rnd,set_i))
        rand_model.save_weights(self.outputFolder+ 'training/model_{}_{}/model'.format(rnd,set_i))
        jsonparams = json.dumps(params)
        with open(self.outputFolder + 'training/model_{}_{}/params.json'.format(rnd,set_i), "w") as outfile:
            outfile.write(jsonparams)
        # writeparams = open(self.outputFolder + 'training/model_{}_{}/params'.format(rnd,set_i),'w')
        # writeparams.write('params\n')
        
        # writeparams.write(params)
       
        # print('Valid loss',valid_loss)
            

        return valid_loss,params
    
    def save_model(self,rnd):
        os.makedirs(self.outputFolder+ 'training/model_{}'.format(rnd))
        self.model.save_weights(self.outputFolder+ 'training/model_{}/model'.format(rnd))


        params = [self.layers,self.neurons,self.activation_list,self.dropout,self.optimizer,self.learning,self.lr_decay,self.factor,self.patience,self.min_lr,self.epochs,self.batch_size]
        jsonparams = json.dumps(params)
        with open(self.outputFolder + 'training/model_{}/params.json'.format(rnd), "w") as outfile:
            outfile.write(jsonparams)

        # np.savetxt(self.outputFolder + 'training/model_{}/params'.format(rnd,),params)
        # ,save_format='tf'

    
    def train(self,rnd):
        inputs, output = self.load_data(rnd,singleRnd=True) 
        self.model,_ = self.surrogate_training(rnd,self.model,inputs,output)
    

    def input_columns_to_training(self,inputs,output=None, unique_inputs=True):
        input_new = []
        output_new = []
        input_non_derivative = []
        j=0
        k=0
        for i in range(np.size(self.input_alias)):
            _,_,derivative_dim,dimension,adjust = self.dict.get_category_values(self.input_alias[i])
            # input[:][i] = (input[:][i]+adjust[0])*adjust[1]
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
        
        if unique_inputs:
            inputs = [input_new,input_new,input_new,input_non_derivative]
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


            return inputs, [np.zeros((np.shape(output_new)[0])),output_new,output_new*0 ]


    def surrogate_training(self,rnd,model,inputs,output,set_i=None,learning_rate='default'):

        if learning_rate == 'default':
            learning_rate = self.learning

        
        inputs,output = self.input_columns_to_training(inputs,output)
        
        inds = np.arange(inputs[0].shape[1])

        if self.WeightRecent == 'Yes':
            # weight the most recent high error points as high as all the other points
            n_points = len(inputs)
            sample_weight = np.ones(n_points)
            if rnd > 0:
                sample_weight[-1000:] = max(1,(n_points-1000)/(2*1000))
            sample_weight = sample_weight[inds]


        # train
        lr_decay = self.lr_decay**rnd
        model.compile(loss=self.lossterms,
                        loss_weights=self.loss_weights,
                        optimizer=eval(self.opt)(learning_rate=learning_rate*lr_decay))
        if set_i==None:
            csv_logger = CSVLogger(self.outputFolder+'training/training_{}.txt'.format(rnd),append=True)
        else:
            csv_logger = CSVLogger(self.outputFolder+'training/training_{}_{}.txt'.format(rnd,set_i),append=True)

        reduceOnPlateau = ReduceLROnPlateau(factor=self.factor,patience=self.patience,min_lr=self.min_lr)
        callbackslist = [csv_logger, reduceOnPlateau]
        if EarlyStopping == 'Yes':
            earlyStopping = EarlyStopping(patience=self.Patience)
            callbackslist.append(earlyStopping)

        # print('Training...')
        # print('model training',model)
        # print(input)
        # print(output)

        print('surrogate training')
        print('inputs',inputs)
        print('outputs',output)

        
        if self.WeightRecent == 'Yes':
            history =model.fit(inputs,
                    output,
                    validation_split=self.validation_split,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    sample_weight=[sample_weight,sample_weight],
                    callbacks=callbackslist)
        
        else:
            history=model.fit(inputs,
                    output,
                    validation_split=self.validation_split,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=callbackslist)
        valid_loss = history.history['val_loss'][-1]
        return model, valid_loss
    
    def IDNN_transforms(self):

        # sys.path.append(self.config_path)
        sys.path.append(self.config_path+self.transform_path)
        from TransformsModule import transforms 

        # return transforms