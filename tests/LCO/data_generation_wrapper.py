#!/usr/bin/env python
#NiAl 

import json
import numpy as np
from subprocess import check_output, STDOUT
import os, shutil, copy
import fileinput as fin
from time import sleep
from shutil import copyfile

def submitCASM(N_jobs,phi,kappa,Tlist,rnd, account, walltime, mem, casm_project_dir='.', test=False, job_manager='slurm', casm_version='LCO', data_generation=[]):

    print(test)
    test=False
    # Calculate and write out the predicted kappa values to the CASM input files
    n = len(kappa)
    phi = np.array(n*[phi])
    if casm_version == 'NiAl':
        kappa = np.expand_dims(kappa,-1).tolist()
        phi = np.expand_dims(phi,-1).tolist()
 
    #kappa = np.expand_dims(kappa,-1).tolist()

    with open(os.path.dirname(__file__)+'/monte_settings_{}.json.tmpl'.format(casm_version),'r') as tmplFile:
             
        tmpl = json.load(tmplFile)
        for job in range(N_jobs):
            # T = Tlist[job]
            shutil.rmtree('job_{}'.format(job+1),ignore_errors=True)
            os.mkdir('job_{}'.format(job+1))

            inputF = copy.deepcopy(tmpl)


            # for T in Tlist:
            for i in range(job,len(kappa),N_jobs):
                phiA = {}
                kappaA = {}
                if casm_version == 'NiAl':
                    inputF['driver']['conditions_list']+=[{'tolerance': 0.001,
                                                    'temperature': T,
                                                    'phi': phi[i],
                                                    'kappa': kappa[i]}]
                elif casm_version == 'LCO':
                    for j in range(len(kappa[0])):
                        phiA[str(j)] = float(phi[i,j])
                        kappaA[str(j)] = float(kappa[i,j])
                    T = Tlist[i]
                    # if ~isinstance(T,str):
                    #     T = str(T)
                    inputF['driver']['custom_conditions']+=[{'tolerance': 0.001,
                                                        'temperature': T,
                                                        'bias_phi': phiA,
                                                        'bias_kappa': kappaA,
                                                        'param_chem_pot': {'a': 0}}]
            # print(inputF)
            # inputF  = str(inputF)
            with open('job_{0}/monte_settings_{0}.json'.format(job+1),'w') as outFile:
                json.dump(inputF,outFile,indent=4)
    string = ""
    for i in data_generation:
        string += str(i) + " "

    # using a data-generation surrogate instead of CASM
    command = []
    if test:
        if job_manager == 'PC':
            for job in range(N_jobs):
                command.append('cd job_{}; python -u {}/data_generation_surrogate_temp.py monte_settings_{}.json {}; cd ../'.format(job+1,os.path.dirname(__file__), job+1, string))
        elif job_manager == 'LSF':
            command = ['cd job_$LSB_JOBINDEX',
                       'python -u {}/data_generation_surrogate_temp.py monte_settings_$LSB_JOBINDEX.json {}'.format(os.path.dirname(__file__), string),
                       'cd ../'] 
        elif job_manager == 'slurm':
            command = ['cd job_$SLURM_ARRAY_TASK_ID',
                       'python -u {}/data_generation_surrogate_temp.py monte_settings_$SLURM_ARRAY_TASK_ID.json {}'.format(os.path.dirname(__file__), string),
                       'cd ../']
    else:
        if job_manager == 'PC':
            raise Exception('JOB_MANAGER cannot be set to PC -- running CASM on PC is not supported. You can run python main_test.py to use a surrogate for data generation')
        elif job_manager == 'LSF':
            command = ['cwd=$PWD',
                    'mv job_$LSB_JOBINDEX {}'.format(casm_project_dir),
                    'cd {}/job_$LSB_JOBINDEX'.format(casm_project_dir),
                    '$CASMPREFIX/bin/casm monte -s monte_settings_$LSB_JOBINDEX.json',
                    'cd ../',
                    'mv job_$LSB_JOBINDEX $cwd']
        elif job_manager == 'slurm':
            print('test')
            command = ['module reset',
                    'module load cpu/0.15.4',
                    'module load anaconda3',
                    '. $ANACONDA3HOME/etc/profile.d/conda.sh',
                    'conda activate /home/jholber/.conda/envs/casm',
                    'cwd=$PWD',
                    'mv job_$SLURM_ARRAY_TASK_ID {}'.format(casm_project_dir),
                    'cd {}/job_$SLURM_ARRAY_TASK_ID'.format(casm_project_dir),
                    #'$CASMPREFIX/bin/casm monte -s monte_settings_$SLURM_ARRAY_TASK_ID.json',
                    'casm monte -s monte_settings_$SLURM_ARRAY_TASK_ID.json',
                    'cd ../',
                    'mv job_$SLURM_ARRAY_TASK_ID $cwd']
    if job_manager == 'PC':
        from subprocess import call
        for job in range(N_jobs):
            call(command[job],shell=True)
    else:
        if job_manager == 'LSF':
            from LSF_manager import submitJob, waitForAll
            specs = {'job_name':'CASM_[1-{}]'.format(N_jobs),
                    'queue': 'gpu_p100',
                    'output_folder':'outputFiles'}
            name = 'CASM*'
            submitJob(command,specs)
            waitForAll(name)
        elif job_manager == 'slurm':
            from slurm_manager import submitJob, waitForAll
            specs = {'job_name':'CASM',
                    'array': '1-{}'.format(N_jobs),
                    'account': account,
                    'walltime': walltime,
                    'total_memory':mem,
                    'output_folder':'outputFiles',
                    'queue': 'shared'}
            name = 'CASM'            
            submitJob(command,specs)
            waitForAll(name)


def compileCASMOutput(rnd, casm_version, len,temp=''):
    kappa = []
    eta = []
    phi = []
    T = []
    # if not os.path.exists(f'round_{rnd}'):
    dirname = "round_"+str(rnd)
    if temp!='':
        dirname = dirname + "_" + str(temp)
    os.mkdir(dirname)
    # else:
    #     os.mkdir(f'round_{rnd}_{temp}')
    for dir in os.listdir('.'):
        if 'job' in dir:
            if os.path.exists(dir+'/results.json'):
                with open(dir+'/results.json','r') as file:
                    data = json.load(file)
                    if casm_version == 'NiAl':
                        kappa += np.array([data['kappa_{}'.format(i)] for i in range(len)]).T.tolist()
                        eta += np.array([data['<op_val({})>'.format(i)] for i in range(len)]).T.tolist()
                        phi += np.array([data['phi_{}'.format(i)] for i in range(len)]).T.tolist()
                    elif casm_version == 'LCO':
                        kappa += np.array([data['Bias_kappa({})'.format(i)] for i in range(len)]).T.tolist()
                        eta += np.array([data['<order_param({})>'.format(i)] for i in range(len)]).T.tolist()
                        phi += np.array([data['Bias_phi({})'.format(i)] for i in range(len)]).T.tolist()
                    T += np.array([data['T']]).T.tolist()

                shutil.move(dir,dirname)

    kappa = np.array(kappa)
    eta = np.array(eta)
    phi = np.array(phi)
    T = np.array(T)
    mu = -2.*phi*(eta - kappa)
    dataOut = np.hstack((kappa,eta,phi,T,mu))
    dataOut = dataOut[~np.isnan(dataOut).any(axis=1)] #remove any rows with nan
    outVars = ['kappa','eta','phi']
    header = ''
    for outVar in outVars:
        for i in range(len):
            header += outVar+'_'+str(i)+' '
    header += 'T '
    for i in range(len):
        header += 'mu_'+str(i)+' '
    np.savetxt('data/results{}.txt'.format(rnd),
               dataOut,
               fmt='%.12f',
               header=header)
    if rnd==0:
        copyfile('data/results{}.txt'.format(rnd),'data/allResults{}.txt'.format(rnd))
    else:
        allResults = np.loadtxt('data/allResults{}.txt'.format(rnd-1))
        allResults = np.vstack((allResults,dataOut))
        np.savetxt('data/allResults{}.txt'.format(rnd),
                   allResults,
                   fmt='%.12f',
                   header=header)

def loadCASMOutput(rnd,dim,singleRnd=False):

    if singleRnd:
        kappa = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float32)[:,:dim]
        eta = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float32)[:,dim:2*dim]
        mu = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float32)[:,-dim:]
        T = np.genfromtxt('data/results'+str(rnd)+'.txt',dtype=np.float32)[:,-dim-1:-dim]
    else:
        kappa = np.genfromtxt('data/allResults'+str(rnd)+'.txt',dtype=np.float32)[:,:dim]
        eta = np.genfromtxt('data/allResults'+str(rnd)+'.txt',dtype=np.float32)[:,dim:2*dim]
        mu = np.genfromtxt('data/allResults'+str(rnd)+'.txt',dtype=np.float32)[:,-dim:]
        T = np.genfromtxt('data/allResults'+str(rnd)+'.txt',dtype=np.float32)[:,-dim-1:-dim]

    return kappa, eta, mu, T

