#!/usr/bin/env python

import os
import io
import fileinput as fin
import shutil
from shutil import copyfile
from operator import itemgetter

from importlib import import_module
from time import sleep

def submitHPSearch(n_sets,rnd,commands,training_func, job_manager, account, walltime, memory):
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

    specs = {'account': account,
             'walltime': walltime,
             'job_name': 'optimizeHParameters',
             'total_memory': memory,
             'queue': 'shared'}
    
    # Compare n_sets of random hyperparameters; choose the set that gives the lowest l2norm
    for i in range(n_sets):
        script = []
        if job_manager != 'PC':
            script.append('python << END')
        script.append('import sys')
        script.append('import numpy as np')
        script.append('rnd = {}'.format(rnd))
        script.append('i = {}'.format(i))
        for command in commands:
            script.append('{}'.format(command))
        script.append('hidden_units, learning_rate, valid_loss = {}(rnd,i)'.format(training_func))
        script.append('if not np.isnan(valid_loss):')
        script.append("\tfout = open('hparameters_{}.txt','w')".format(i))
        script.append("\tfout.write('hparameters += [[{},{},\"{}_{}\",{}]]'.format(learning_rate,hidden_units,rnd,i,valid_loss))")
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
            from slurm_manager import numCurrentJobs, submitJob
            submitJob(script,specs)

def hyperparameterSearch(rnd,N_sets,commands,training_func,job_manager, account, walltime, memory):
    """ A function that initializes and manages the hyperparameter search in the active learning workflow.

    (Still needs to be generalized).

    :param N_sets: The number of hyperparameter sets to run.
    :type N_sets: int

    :param rnd: The current round (workflow iteration) number.
    :type rnd: int
    
    """
    
    # Submit the training sessions with various hyperparameters
    submitHPSearch(N_sets,rnd,commands,training_func, job_manager, account, walltime, memory)

    # Wait for jobs to finish
    if job_manager != 'PC':
        from slurm_manager import numCurrentJobs, submitJob
        sleep(20)
        while ( numCurrentJobs('optimizeHParameters') > 0):
            sleep(15)

    # Compare n_sets of random hyperparameters; choose the set that gives the lowest l2norm
    hparameters = []
    for i in range(N_sets):
        filename = 'hparameters_'+str(i)+'.txt'
        if os.path.isfile(filename):
            fin = open(filename,'r')
            exec (fin.read()) # execute the code snippet written as a string in the read file
            fin.close()
            os.remove('hparameters_'+str(i)+'.txt')

    # Sort by l2norm
    sortedHP = sorted(hparameters,key=itemgetter(3))

    writeHP = open('data/sortedHyperParameters_'+str(rnd)+'.txt','w')
    writeHP.write('learning_rate,hidden_units,round/set,l2norm\n')
    for set in sortedHP:
        writeHP.write(str(set[0])+','+str(set[1])+',"'+str(set[2])+'",'+str(set[3])+'\n')
    writeHP.close()

    # Clean up checkpoint files
    #os.rename('idnn_{}_{}.h5'.format(rnd,sortedHP[0][2]),'idnn_{}.h5'.format(rnd))
    shutil.rmtree('idnn_{}'.format(rnd),ignore_errors=True)
    os.rename('idnn_{}'.format(sortedHP[0][2]),'idnn_{}'.format(rnd))
    copyfile('training/training_{}.txt'.format(sortedHP[0][2]),'training/training_{}.txt'.format(rnd))
    for i in range(N_sets):
        shutil.rmtree('idnn_{}_{}'.format(rnd,i),ignore_errors=True)

    return sortedHP[0][1],sortedHP[0][0] #hidden_units, learning_rate
