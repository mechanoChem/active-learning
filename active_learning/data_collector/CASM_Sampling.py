import json
import numpy as np
from subprocess import check_output, STDOUT
import os, shutil, copy
import fileinput as fin
from time import sleep
from shutil import copyfile
from active_learning.data_collector.sampling import Sampling
import pandas as pd
from active_learning.data_collector.slurm_manager import submitJob, waitForAll

class CASM_Sampling(Sampling):    

    def __init__(self,model,dictionary): 
        ## determine dictionary ie 
        super().__init__(model,dictionary)

        self.model = model
        self.dict = dictionary 
        [self.data_gen_source] = self.dict.get_individual_keys('Main',['data_generation_source'])

        if self.data_gen_source == 'CASM':
            [self.dir, self.version, self.Initial_mu,self.phi,self.N_jobs,self.relevent_indices] = self.dict.get_category_values('CASM')
        else:
            [self.dir, self.version, self.Initial_mu,self.phi,self.N_jobs,self.Hidden_Layers, self.Input_Shape, self.dim, self.CASM_version, self.data_gen_activation, self.folder,self.relevent_indices] = self.dict.get_category_values('CASM_Surrogate')

        [self.Model_type,    
         self.Data_Generation, self.Data_Generation_Source, self.restart,
         self.input_data,self.input_alias,self.output_alias, self.iterations,
         self.OutputFolder, self.Input_dim, self.derivative_dim, self.output_dim,_,self.T,self.testing_set,self.graph,_,_,_,_] = self.dict.get_category_values('Main')
        
        [self.job_manager,self.account,self.walltime,self.mem] = self.dict.get_category_values('Sampling_Job_Manager')
        
        self.sampling_dict = self.dict.get_category('Sampling')

        [self.layers,self.neurons,self.activation,self.dropout,self.transform_path,
         self.lossterms,self.loss_weights,self.optimizer,self.learning,self.lr_decay,
         self.factor, self.patience,self.min_lr,self.EarlyStopping,self.epochs,
         self.batch_size,self.WeightRecent,self.validation_split,
         self.hyperparameter,self.train_new_idnn] = self.dict.get_category_values('IDNN')

        for i in range(np.size(self.lossterms)):
            if self.lossterms[i] == 'None':
                self.lossterms[i] = None

        self.global_database=False
        name = f'CASM_{self.OutputFolder}' 
        if self.job_manager == 'slurm':
            waitForAll(name)


    def load_single_rnd_output(self,rnd):
        kappa = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float64)[:,:self.derivative_dim]
        eta = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float64)[:,self.derivative_dim:2*self.derivative_dim]
        mu = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float64)[:,-self.derivative_dim:]
        T = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float64)[:,-self.derivative_dim-1:-self.derivative_dim]
        return eta,mu,T
    
    def read(self,rnd,singleRnd=True):
        read_data = np.genfromtxt(self.OutputFolder+'data/data_recommended/rnd'+str(rnd)+'.txt',dtype=np.float64)
        if self.global_database:
            #if self.global_database - exclude billiardwalk points ie value =1
            return read_data[read_data[-1]!=1, :]
        else:
            return read_data

    def save(self,output,rnd,type):
        np.savetxt(self.OutputFolder+'data/data_recommended/'+type+'_rnd'+str(rnd)+'.txt',output,fmt='%.12f')

    def read_from_casm(self,rnd,existingglobal=False,num0=0,num1=0,explore2D=False,points_2D=False):
        # rows_to_keep = [0, 29, 30, 31]
        # rows_to_keep = [0,1,2,3,4,5,6]
        number = len(self.relevent_indices)
        kappa = []
        eta = []
        phi = []
        T = []
        mu = []
        # if not os.path.exists(f'round_{rnd}'):
        dirname = self.OutputFolder + 'data/data_sampled/round_'+str(rnd)
        # shutil.rmtree(dirname,ignore_errors=True)
        os.mkdir(dirname)
        data_points = []
        print("Reading from CASM")
        # else:
        #     os.mkdir(f'round_{rnd}_{temp}')
        for dir in os.listdir(self.OutputFolder + 'data/data_sampled/'):
            if 'job' in dir:
                if os.path.exists(self.OutputFolder + 'data/data_sampled/'+dir+'/results.json'):
                    with open(self.OutputFolder + 'data/data_sampled/'+dir+'/results.json','r') as file:
                        try:
                            data = json.load(file)
                            if self.version == 'NiAl':
                                kappa += np.array([data['kappa_{}'.format(i)] for i in range(self.Input_dim-1)]).T.tolist()
                                eta += np.array([data['<op_val({})>'.format(i)] for i in range(self.Input_dim-1)]).T.tolist()
                                phi += np.array([data['phi_{}'.format(i)] for i in range(self.Input_dim-1)]).T.tolist()
                            elif self.version == 'LCO' or self.data_gen_source=='CASM_Surrogate':
                                kappa += np.array([data['Bias_kappa({})'.format(i)] for i in range(self.Input_dim-1)]).T.tolist()
                                eta += np.array([data['<order_param({})>'.format(i)] for i in range(self.Input_dim-1)]).T.tolist()
                                phi += np.array([data['Bias_phi({})'.format(i)] for i in range(self.Input_dim-1)]).T.tolist()
                                mu += np.array([data['<mu({})>'.format(i)] for i in range(self.Input_dim-1)]).T.tolist()
                                T += np.array([data['T']]).T.tolist()
                            elif self.version == 'row':    
                                monte_path=''
                                directory = os.path.join(self.OutputFolder + 'data/data_sampled/', dir)
                            
                                for filename in os.listdir(directory):
                                    if 'monte' in filename:
                                        monte_path = os.path.join(directory, filename)
                                if monte_path !='':
                                    with open(monte_path,'r') as file:
                                        monte_file = json.load(file)
                                        driver= monte_file['driver']
                                        conditions = driver['custom_conditions']
                                        for i in range(len(data['<comp(a)>'])):
                                            conditions_data = conditions[i]
                                            converged = data['is_converged']

                                            kappa = conditions_data['order_parameter_quad_pot_target']
                                            phi = conditions_data['order_parameter_quad_pot_vector']
                                            phi_subset=number*[0]
                                            kappa_subset=number*[0]
                                            mu = number*[0]
                                            eta = number*[0]
                                            # eta0 = c
                                            k=0
                                            for j in self.relevent_indices:
                                                eta[k] = data['<order_parameter({})>'.format(j)][i]
                                                mu[k] =(-2*phi[j]*(eta[k]-kappa[j]))*32
                                                phi_subset[k] = phi[j]*32*32
                                                kappa_subset[k] = kappa[j]/32
                                                T = conditions_data['temperature']
                                                eta[k] = eta[k]/32
                                                k+=1
                                            if converged[i]==True:
                                                data_points.append({
                                                    'kappa': kappa_subset,
                                                    'phi': phi_subset,
                                                    'mu': mu,
                                                    'T': T,
                                                    'eta': eta,
                                                })
                        except:
                            print('Did not use',dir)


                    shutil.move(self.OutputFolder + 'data/data_sampled/'+dir,dirname)


        if self.version == 'row' and self.data_gen_source!='CASM_Surrogate':
            kappa = np.array([d['kappa'] for d in data_points])
            eta = np.array([d['eta'] for d in data_points])
            phi = np.array([d['phi'] for d in data_points])
            T = np.array([d['T'] for d in data_points])
            mu= np.array([d['mu'] for d in data_points])
            T = np.reshape(T,(len(T),1))
        elif self.version == 'row' and self.data_gen_source=='CASM_Surrogate':
            kappa = np.array(kappa,dtype=float)
            eta = np.array(eta,dtype=float)
            phi = np.array(phi,dtype=float)
            T = np.array(T,dtype=float)[0,:,:]
            mu = np.array(mu,dtype=float)
        else:
            kappa = np.array(kappa,dtype=float)
            eta = np.array(eta,dtype=float)
            phi = np.array(phi,dtype=float)
            T = np.array(T,dtype=float)[0,:,:]
            mu = -2.*phi*(eta - kappa)

        if existingglobal:
            points2dlen=0



            if explore2D:
                print('Adding Billiard points + 2D')
                billardpoints = np.genfromtxt('shuffled.txt',dtype=np.float64)[round(num0*(1-points_2D)):round(num1*(1-points_2D)),:]
                self.save(billardpoints,rnd,'billiardwalk_preexisting')
                billardpoints2D = np.genfromtxt('billiard2D.txt',dtype=np.float64)[round(num0*points_2D):round(num1*points_2D),:]
                self.save(billardpoints2D,rnd,'billiardwalk2D_preexisting')
                points2dlen=np.shape(billardpoints2D)[0]
                billardpoints = np.vstack((billardpoints,billardpoints2D))

            else:
                print('Adding Billiard points')
                billardpoints = np.genfromtxt('shuffled.txt',dtype=np.float64)[num0:num1,:]
                self.save(billardpoints,rnd,'billiardwalk_preexisting')
            
            originallen = np.shape(kappa)[0]
            kappa = np.vstack((kappa,billardpoints[:,:self.derivative_dim]))
            eta = np.vstack((eta,billardpoints[:,self.derivative_dim:2*self.derivative_dim]))
            phi= np.vstack((phi,billardpoints[:,2*self.derivative_dim:3*self.derivative_dim]))
            T = np.vstack((T,billardpoints[:,-self.derivative_dim-1:-self.derivative_dim]) )
            mu =np.vstack((mu,billardpoints[:,-self.derivative_dim:] ))
            output = np.hstack((kappa,T, np.zeros(np.shape(T))))
            output = output[originallen:,:]
            if points2dlen !=0:
                output[-points2dlen:,-1] = 1
        else:
            output =None



            
            # billardpoints = np.genfromtxt('billiardwalkpoints.txt',dtype=np.float64)[self.pointcount:self.pointcount+self.N_global_pts,:]      
        self.write(rnd,kappa,eta,phi,T,mu)
        return output
    

    def write(self,rnd,kappa,eta,phi,T,mu):

        dataOut = np.hstack((kappa,eta,phi,T,mu))
        dataOut = dataOut[~pd.isna(dataOut).any(axis=1)] #remove any rows with nan



        outVars = ['kappa','eta','phi']
        header = ''
        for outVar in outVars:
            for i in range(self.Input_dim-1):
                header += outVar+'_'+str(i)+' '
        header += 'T '
        for i in range(self.Input_dim-1):
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



        dataOut = np.hstack((eta,T,mu))



        np.save(self.OutputFolder + 'data/data_sampled/results{}'.format(rnd),
        dataOut)
        if rnd==0:
            np.save(self.OutputFolder + 'data/data_sampled/allResults{}'.format(rnd),
                    dataOut)
        else:
            allResults =  np.load(self.OutputFolder + 'data/data_sampled/allResults{}.npy'.format(rnd-1),allow_pickle=True)
            allResults = np.vstack((allResults,dataOut))

            np.save(self.OutputFolder + 'data/data_sampled/allResults{}'.format(rnd),
                    allResults)
        
        np.savetxt(self.OutputFolder + 'data/data_sampled/results{}.txt'.format(rnd),dataOut)
        
            


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
        if rnd<self.Initial_mu:
            mu_test = np.zeros(np.shape(eta))
        else:
            mu_test = self.model.predict([eta,T])[1]

        kappa = eta + 0.5*mu_test/self.phi

        n = len(kappa)
        phi = np.array(n*[self.phi])
        
        if self.version == 'NiAl':
            kappa = np.expand_dims(kappa,-1).tolist()
            phi = np.expand_dims(phi,-1).tolist()

     
        with open('{}/monte_settings_zigzag.json.tmpl'.format(os.path.dirname(__file__)),'r') as tmplFile:
                
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
                        TA = float(T[i,0])
       
                        inputF['driver']['custom_conditions']+=[{'tolerance': .005,
                                                            'temperature': TA,
                                                            'bias_phi': phiA,
                                                            'bias_kappa': kappaA,
                                                            'param_chem_pot': {'a': 0}}]
                    
                    elif self.version == 'row':
                        
                        kappas = [0]*32
                        k=0
                        for j in self.relevent_indices:
                            kappas[j] = kappa[i,k]*32
                            k+=1
                        TA = float(T[i,0])

                        maxvalue = max(np.abs(kappa[i,1]),np.abs(kappa[i,2]),np.abs(kappa[i,3])  )
                        phiA = [0]*32
                        phiA[0] = 0.009765625*5
                        for x in range(1,32):
                            phiA[x]=0.00009765625*.5


                        inputF['driver']['custom_conditions']+=[{'tolerance': 0.001,
                                    'temperature': TA,
                                    'order_parameter_quad_pot_vector': phiA,
                                    'order_parameter_quad_pot_target': kappas}]
        
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
                command = [f'cd {self.OutputFolder}/data/data_sampled/job_$SLURM_ARRAY_TASK_ID',
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
                        # 'conda activate /home/jholber/.local/conda/envs/CASM',
                        'conda activate /home/jholber/.local/conda3/envs/casm_python3_11', 
                        "export LIBCASM='/home/jholber/.local/conda3/envs/casm_python3_11/lib/libcasm.so'",
                        'cwd=$PWD',
                        'mv {}data/data_sampled/job_$SLURM_ARRAY_TASK_ID {}'.format(self.OutputFolder,self.dir),
                        'cd {}/job_$SLURM_ARRAY_TASK_ID'.format(self.dir),
                        #'$CASMPREFIX/bin/casm monte -s monte_settings_$SLURM_ARRAY_TASK_ID.json',
                        'ccasm monte -s monte_settings_$SLURM_ARRAY_TASK_ID.json',
                        'cd ../',
                        'mv job_$SLURM_ARRAY_TASK_ID $cwd/{}data/data_sampled/'.format(self.OutputFolder)]
        return command

    import os


    def move_subfolders(self,source_dir, destination_dir):
        # Ensure the destination directory exists
        os.makedirs(destination_dir, exist_ok=True)

        # Iterate through the items in the source directory
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)

            # Check if the item is a directory
            if os.path.isdir(item_path):
                # Construct the destination path
                dest_path = os.path.join(destination_dir, item)

                # Move the subfolder to the new folder
                shutil.move(item_path, dest_path)
                print(f"Moved: {item_path} -> {dest_path}")



    def submit_job(self,command):
        if self.job_manager == 'PC':
            from subprocess import call
            for job in range(self.N_jobs):
                call(command[job],shell=True)
        else:
            if self.job_manager == 'slurm':
                from active_learning.data_collector.slurm_manager import submitJob, waitForAll
                name = f'CASM_{self.OutputFolder}' 
                specs = {'job_name':name,
                        'array': '1-{}'.format(self.N_jobs),
                        'account': self.account,
                        'wall_time': self.walltime,
                        'total_memory':self.mem,
                        'output_folder':'outputFiles',
                        'queue': 'shared'}   
                print('about to submit job')        
                submitJob(command,specs)
                waitForAll(name)
                # Example usage
                source_directory = self.dir
                destination_directory = self.OutputFolder+ 'data/data_sampled/'

                self.move_subfolders(source_directory, destination_directory)




