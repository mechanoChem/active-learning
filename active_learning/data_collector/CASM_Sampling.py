import json
import numpy as np
from subprocess import check_output, STDOUT
import os, shutil, copy
import fileinput as fin
from time import sleep
from shutil import copyfile
from active_learning.data_collector.Sampling import Sampling
import pandas as pd

class CASM_Sampling(Sampling):    

    def __init__(self,model,dictionary): 
        ## determine dictionary ie 
        super().__init__(model,dictionary)
        self.model = model
        self.dict = dictionary 
        [self.data_gen_source] = self.dict.get_individual_keys('Main',['data_generation_source'])

        if self.data_gen_source == 'CASM':
            [self.dir, self.version, self.Initial_mu,self.phi,self.N_jobs] = self.dict.get_category_values('CASM')
        else:
            [self.dir, self.version, self.Initial_mu,self.phi,self.N_jobs,self.Hidden_Layers, self.Input_Shape, self.dim, self.CASM_version, self.data_gen_activation, self.folder] = self.dict.get_category_values('CASM_Surrogate')
            # [self.Hidden_Layers, self.Input_Shape, self.dim, self.CASM_version, self.data_gen_activation, self.folder]=self.dict.get_category_values('CASM Surrogate')

        # print('hidden layers',self.Hidden_Layers)
        [self.Model_type,    
         self.Data_Generation, self.Data_Generation_Source, self.restart,
         self.input_data,self.input_alias,self.output_alias, self.iterations,
         self.OutputFolder, self.seed, self.Input_dim, self.derivative_dim, self.output_dim,_] = self.dict.get_category_values('Main')
        
        [self.job_manager,self.account,self.walltime,self.mem] = self.dict.get_category_values('Sampling_Job_Manager')
        
        self.sampling_dict = self.dict.get_category('Sampling')

        # print('input dim',self.Input_dim)



    def load_single_rnd_output(self,rnd):
        kappa = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float32)[:,:self.derivative_dim]
        eta = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float32)[:,self.derivative_dim:2*self.derivative_dim]
        mu = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float32)[:,-self.derivative_dim:]
        T = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float32)[:,-self.derivative_dim-1:-self.derivative_dim]
        return eta,mu,T
    
    def read(self,rnd,singleRnd=True):
        return np.genfromtxt(self.OutputFolder+'data/data_recommended/rnd'+str(rnd)+'.txt',dtype=np.float32)

    def write(self,rnd):
        kappa = []
        eta = []
        phi = []
        T = []
        # if not os.path.exists(f'round_{rnd}'):
        dirname = self.OutputFolder + 'data/data_sampled/round_'+str(rnd)
        shutil.rmtree(dirname,ignore_errors=True)
        os.mkdir(dirname)
        # else:
        #     os.mkdir(f'round_{rnd}_{temp}')
        for dir in os.listdir(self.OutputFolder + 'data/data_sampled/'):
            if 'job' in dir:
                if os.path.exists(self.OutputFolder + 'data/data_sampled/'+dir+'/results.json'):
                    with open(self.OutputFolder + 'data/data_sampled/'+dir+'/results.json','r') as file:
                        data = json.load(file)
                        if self.version == 'NiAl':
                            kappa += np.array([data['kappa_{}'.format(i)] for i in range(self.Input_dim-1)]).T.tolist()
                            eta += np.array([data['<op_val({})>'.format(i)] for i in range(self.Input_dim-1)]).T.tolist()
                            phi += np.array([data['phi_{}'.format(i)] for i in range(self.Input_dim-1)]).T.tolist()
                        elif self.version == 'LCO':
                            kappa += np.array([data['Bias_kappa({})'.format(i)] for i in range(self.Input_dim-1)]).T.tolist()
                            eta += np.array([data['<order_param({})>'.format(i)] for i in range(self.Input_dim-1)]).T.tolist()
                            phi += np.array([data['Bias_phi({})'.format(i)] for i in range(self.Input_dim-1)]).T.tolist()
                        T += np.array([data['T']]).T.tolist()

                    shutil.move(self.OutputFolder + 'data/data_sampled/'+dir,dirname)

        kappa = np.array(kappa,dtype=float)
        eta = np.array(eta,dtype=float)
        phi = np.array(phi,dtype=float)
        T = np.array(T,dtype=float)
        mu = -2.*phi*(eta - kappa)
        dataOut = np.hstack((kappa,eta,phi,T,mu))
        dataOut = dataOut[~pd.isna(dataOut).any(axis=1)] #remove any rows with nan
        outVars = ['kappa','eta','phi']
        header = ''
        for outVar in outVars:
            for i in range(self.Input_dim):
                header += outVar+'_'+str(i)+' '
        header += 'T '
        for i in range(self.Input_dim):
            header += 'mu_'+str(i)+' '

        np.savetxt(self.OutputFolder + 'data/data_sampled/CASMresults{}.txt'.format(rnd),
                dataOut,
                header=header)
        if rnd==0:
            copyfile(self.OutputFolder + 'data/data_sampled/CASMresults{}.txt'.format(rnd),self.OutputFolder + 'data/data_sampled/CASMallResults{}.txt'.format(rnd))
        else:
            allResults = np.loadtxt(self.OutputFolder + 'data/data_sampled/CASMallResults{}.txt'.format(rnd-1))
            allResults = np.vstack((allResults,dataOut))
            np.savetxt(self.OutputFolder + 'data/data_sampled/CASMallResults{}.txt'.format(rnd),
                    allResults,
                    header=header)
        

        eta = eta.T
        mu = mu.T
        T = T.T


        input_derivative = np.array([np.zeros(np.shape(eta)),eta,eta*0])
        input_non_derivative = np.array([T])
        output =[0*T, mu, 0*mu]
        

        dataOut = np.array([input_derivative,input_non_derivative,output], dtype=object)
        np.save(self.OutputFolder + 'data/data_sampled/results{}'.format(rnd),
        dataOut)
        if rnd==0:
            np.save(self.OutputFolder + 'data/data_sampled/allResults{}'.format(rnd),
                    dataOut)
        else:
            allResults =  np.load(self.OutputFolder + 'data/data_sampled/results{}.npy'.format(rnd-1),allow_pickle=True)
            for i in range(2):
                # print('CASM sampling line 115')
                # print(np.shape(allResults[i]))
                # print(np.shape(dataOut[i]))
                allResults[i] = np.concatenate((allResults[i],dataOut[i]),axis=-1)
                # print(np.shape(allResults[i]))
            # print('CASM sampling line 117')

            for j in range(3):
                # print(np.shape(allResults[2][j]))
                # print(np.shape(dataOut[2][j]))
                allResults[2][j] = np.concatenate((allResults[2][j],dataOut[2][j]),axis=-1)
                # print(np.shape(allResults[2][j]))


            np.save(self.OutputFolder + 'data/data_sampled/allResults{}'.format(rnd),
                    allResults)
        
            


    def ideal(self,x_test,T):
        kB = 8.61733e-5 
        invQ = self.sampling_dict['continuous_dependent']['Qmatrix']['invQ']
        mu_test = 0.25*kB*np.log(x_test/(1.-x_test)).dot(invQ.T)
        mu_test = np.multiply(T,mu_test)

        return mu_test

    
    def construct_job(self,rnd):
        data_recommended = self.read(rnd) 
        eta = data_recommended[:,:self.Input_dim-1]
        T = data_recommended[:,self.Input_dim-1:self.Input_dim]
        if rnd==0:
            if self.Initial_mu == 'Ideal':
                mu_test = self.ideal(eta,T)
            else:
                mu_test = np.zeros(np.shape(eta))
        else:
            mu_test = self.model.predict([eta,eta,eta,T])[1]


        kappa = eta + 0.5*mu_test/self.phi

        n = len(kappa)
        phi = np.array(n*[self.phi])
        if self.version == 'NiAl':
            kappa = np.expand_dims(kappa,-1).tolist()
            phi = np.expand_dims(phi,-1).tolist()

        with open(os.path.dirname(__file__)+'/monte_settings_{}.json.tmpl'.format(self.version),'r') as tmplFile:
                
            tmpl = json.load(tmplFile)
            for job in range(self.N_jobs):
                shutil.rmtree(self.OutputFolder + 'data/data_sampled/job_{}'.format(job+1),ignore_errors=True)
                os.mkdir(self.OutputFolder + 'data/data_sampled/job_{}'.format(job+1))

                inputF = copy.deepcopy(tmpl)


                for i in range(job,len(kappa),self.N_jobs):
                    phiA = {}
                    kappaA = {}
                    if self.version == 'NiAl':
                        inputF['driver']['conditions_list']+=[{'tolerance': 0.001,
                                                        'temperature': T[i],
                                                        'phi': phi[i],
                                                        'kappa': kappa[i]}]
                    elif self.version == 'LCO':
                        for j in range(len(kappa[0])):
                            phiA[str(j)] = float(phi[i,j])
                            kappaA[str(j)] = float(kappa[i,j])
                        TA = T[i,0]
                        if ~isinstance(TA,str):
                            TA = str(TA)
                        inputF['driver']['custom_conditions']+=[{'tolerance': 0.001,
                                                            'temperature': TA,
                                                            'bias_phi': phiA,
                                                            'bias_kappa': kappaA,
                                                            'param_chem_pot': {'a': 0}}]

                with open(self.OutputFolder + 'data/data_sampled/job_{0}/monte_settings_{0}.json'.format(job+1),'w') as outFile:
                    json.dump(inputF,outFile,indent=4)


        command = []
        if self.Data_Generation_Source == 'CASM_Surrogate':
            data_generation = [self.Hidden_Layers, self.Input_Shape, self.dim, self.CASM_version, self.data_gen_activation, self.folder]
            string = ""
            for i in data_generation:
                string += str(i) + " "
            if self.job_manager == 'PC':
                for job in range(self.N_jobs):
                    command.append('cd '+ self.OutputFolder + 'data/data_sampled/job_{}; python -u {}/data_generation_surrogate_temp.py monte_settings_{}.json {}; cd ../'.format(job+1,os.path.dirname(__file__), job+1, string))
            elif self.job_manager == 'LSF':
                command = ['cd job_$LSB_JOBINDEX',
                        'python -u {}/data_generation_surrogate_temp.py monte_settings_$LSB_JOBINDEX.json {}'.format(os.path.dirname(__file__), string),
                        'cd ../'] 
            elif self.job_manager == 'slurm':
                command = ['cd job_$SLURM_ARRAY_TASK_ID',
                        'python -u {}/data_generation_surrogate_temp.py monte_settings_$SLURM_ARRAY_TASK_ID.json {}'.format(os.path.dirname(__file__), string),
                        'cd ../']

        else:
            if self.job_manager == 'PC':
                raise Exception('JOB_MANAGER cannot be set to PC -- running CASM on PC is not supported. You can run python main_test.py to use a surrogate for data generation')
            elif self.job_manager == 'LSF':
                command = ['cwd=$PWD',
                        'mv job_$LSB_JOBINDEX {}'.format(self.dir),
                        'cd {}/job_$LSB_JOBINDEX'.format(self.dir),
                        '$CASMPREFIX/bin/casm monte -s monte_settings_$LSB_JOBINDEX.json',
                        'cd ../',
                        'mv job_$LSB_JOBINDEX $cwd']
            elif self.job_manager == 'slurm':
                command = ['module reset',
                        'module load cpu/0.15.4',
                        'module load anaconda3',
                        '. $ANACONDA3HOME/etc/profile.d/conda.sh',
                        'conda activate /home/jholber/.conda/envs/casm',
                        'cwd=$PWD',
                        'mv job_$SLURM_ARRAY_TASK_ID {}'.format(self.dir),
                        'cd {}/job_$SLURM_ARRAY_TASK_ID'.format(self.dir),
                        #'$CASMPREFIX/bin/casm monte -s monte_settings_$SLURM_ARRAY_TASK_ID.json',
                        'casm monte -s monte_settings_$SLURM_ARRAY_TASK_ID.json',
                        'cd ../',
                        'mv job_$SLURM_ARRAY_TASK_ID $cwd']
        return command

    def submit_job(self,command):
        if self.job_manager == 'PC':
            from subprocess import call
            for job in range(self.N_jobs):
                call(command[job],shell=True)
        else:
            if self.job_manager == 'slurm':
                from slurm_manager import submitJob, waitForAll
                specs = {'job_name':'CASM',
                        'array': '1-{}'.format(self.N_jobs),
                        'account': self.account,
                        'walltime': self.walltime,
                        'total_memory':self.mem,
                        'output_folder':'outputFiles',
                        'queue': 'shared'}
                name = 'CASM'            
                submitJob(command,specs)
                waitForAll(name)


