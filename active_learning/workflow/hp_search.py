#!/usr/bin/env python

import os
import io
import fileinput as fin
import shutil
from shutil import copyfile
from operator import itemgetter

from importlib import import_module
from time import sleep
import sys

def submitHPSearch(n_sets,rnd,commands,training_func, job_manager, account, walltime, memory,label,outputfolder,original=True):
    """ A function to submit the job scripts for a each set of hyperparameters
    in the hyperparameter search in the active learning workflow.

    (Still needs to be generalized).

    :param n_sets: The number of hyperparameter sets to run.
    :type n_sets: int

    :param rnd: The current round (workflow iteration) number.
    :type rnd: int
    
    """


    if job_manager=='PC':
        from subprocess import call

    if original:
        rangevals=[0,n_sets]
    else:
        rangevals = [n_sets,2*n_sets]
    
    # Compare n_sets of random hyperparameters; choose the set that gives the lowest l2norm
    for i in range(n_sets):
        specs = {'account': account,
            'wall_time': walltime,
            'job_name': 'optHPparam_{}'.format(label),
            'total_memory': memory,
            'queue': 'shared'}

        script = []
        if job_manager != 'PC':
            script.append('python << END')
        script.append('import sys')
        script.append('import numpy as np')
        # script.append('sys.path.append({})'.format(sys.path[0]))
        script.append('rnd = {}'.format(rnd))
        script.append('i = {}'.format(i))
        for command in commands:
            script.append('{}'.format(command))
        script.append('valid_loss,params = {}(rnd,i)'.format(training_func))
        script.append('if not np.isnan(valid_loss):')
        script.append("\tfout = open('{}hparameters_{}.txt','w')".format(outputfolder,i))
        script.append("\tfout.write('hparameters += [[{},\"{}_{}\",{}]]'.format(params,rnd,i,valid_loss))")
        script.append('\tfout.close()')
        if job_manager != 'PC':
            script.append('END')
        if job_manager == 'PC':
            #output = io.StringIO()
            textfile = open("optimize_hparameters.py", "w")
            for element in script:
                #output.write(element + "\n")
                textfile.write(element + "\n")
            textfile.close()
           # call('python' + output.read(), shell=True)
            call('python '+ 'optimize_hparameters.py', shell=True)
            #call('python ' + str(script), shell=True)
        else:      
            from active_learning.data_collector.slurm_manager import numCurrentJobs, submitJob
            submitJob(script,specs,num=i,slurmdirectory=outputfolder+'/slurm/')

def hyperparameterSearch(rnd,n_sets,commands,training_func,job_manager, account, walltime, memory,outputfolder,model_loss,original=True):
    """ A function that initializes and manages the hyperparameter search in the active learning workflow.

    (Still needs to be generalized).

    :param n_sets: The number of hyperparameter sets to run.
    :type n_sets: int

    :param rnd: The current round (workflow iteration) number.
    :type rnd: int
    
    """
    
    # Submit the training sessions with various hyperparameters
    i=0
    print('job_manager',job_manager)
    if job_manager != 'PC':
        from active_learning.data_collector.slurm_manager import numCurrentJobs, submitJob
        while numCurrentJobs('optHPparam_{}'.format(i)) > 0:
            i+=1


    submitHPSearch(n_sets,rnd,commands,training_func, job_manager, account, walltime, memory,i,outputfolder,original)

    if job_manager != 'PC':
        from active_learning.data_collector.slurm_manager import numCurrentJobs, submitJob
        sleep(20)
        while ( numCurrentJobs('optHPparam_{}'.format(i)) > 0):
            sleep(15)

    # Compare n_sets of random hyperparameters; choose the set that gives the lowest l2norm
    hparameters = []
    if original:
        rangevals=[0,n_sets]
    else:
        rangevals = [n_sets,2*n_sets]
    
    # Compare n_sets of random hyperparameters; choose the set that gives the lowest l2norm
    for i in range(n_sets):
        filename = outputfolder+'hparameters_'+str(i)+'.txt'
        # print('hparameters location',filename)
        if os.path.isfile(filename):
            fin = open(filename,'r')
            exec (fin.read()) # execute the code snippet written as a string in the read file
            fin.close()
            os.remove(outputfolder+'hparameters_'+str(i)+'.txt')

    # Sort by l2norm
    print('hparameters: ', hparameters)
    sortedHP2 = sorted(hparameters,key=itemgetter(1))
    print('sortedhp2',sortedHP2)
    sortedHP = sorted(hparameters,key=itemgetter(2))
    print('sortedhp',sortedHP)
    # self.outputFolder+ 'training/

    if original:
        writeHP = open(outputfolder + 'training/trainings/sortedHyperParameters_'+str(rnd)+'.txt','w')
    else:
        writeHP = open(outputfolder + 'training/trainings/sortedHyperParameters_'+str(rnd)+'_2.txt','w')
    writeHP.write('params,round/set,l2norm\n')
    for set in sortedHP:
        writeHP.write(str(set[0])+','+str(set[1])+',"'+str(set[2])+'\n')
    writeHP.close()

    # Clean up checkpoint files
    #os.rename('idnn_{}_{}.h5'.format(rnd,sortedHP[0][2]),'idnn_{}.h5'.format(rnd))
    print('sortedHP',sortedHP)
    print('sortedHp[0][-1]',sortedHP[0][-1])
    print('type sorted', type(sortedHP[0][-1]))
    print('model loss',model_loss)
    print('type loss',type(model_loss))
     # Compare n_sets of random hyperparameters; choose the set that gives the lowest l2norm
    # if sortedHP[0][-1] < model_loss:
    if 1==1:
        shutil.rmtree(outputfolder + 'training/model_{}'.format(rnd),ignore_errors=True)
        # print('sortedhp', sortedHP)
        os.rename(outputfolder + 'training/model_{}'.format(sortedHP[0][1],rnd),outputfolder + 'training/model_{}'.format(rnd))
        copyfile(outputfolder +'training/trainings/training_{}.txt'.format(sortedHP[0][1]),outputfolder +'training/training_{}.txt'.format(rnd))
    

    for i in range(n_sets):
        shutil.rmtree(outputfolder + 'training/model_{}_{}'.format(rnd,i),ignore_errors=True)

    return sortedHP[0][0],sortedHP[0][-1] #,sortedHP[0][0] #params
