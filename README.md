# Active Learning

This repository provides an Active Learning Customizable Workflow for the training of a model and the recommendation of data to sample based on the model training. 

The workflow developed here can train a model, update model parameters, recommend data to sample, and collect the data. Currently the workflow includes an IDNN model, hyperpameter searches, monte carlo calculations, and data recommendations based on sobol sequence, billiardwalk, random space sampling, samping near wells, sampling near vertices, high error points, finding wells and sampling, and finding non-convexities and sampling. 

The workflow is readily adapatable for a variety of problems, and could be used for different models and different sampling mechanisms, including working with experimental results. 

Main directories:
* active_learning
  * data_collector: collects data, currently compatible with CASM or a CASM surrogate
  * data_recommended: recommendes data to sample using both explorative and exploitative methods
  * model: surrogate model to train, currently compatabile with an IDNN
  * workflow: set of code that controls the workflow
* tests
  * LCO: example set of input files for running the code using an IDNN and CASM sampling   



![alt text](https://github.com/mechanoChem/active-learning/blob/main/active_learning/workflow/active_learning_general.png "Overview of Workflow")


# Installation and Running of Code

## Installation and setup

    git clone https://github.com/mechanoChem/active-learning.git
    pip install matplotlib
                tensorflow
                sobol-seq
                pandas


## To run the example

    cd active_learning/tests/LCO/
    python main.py input.ini

## To run your own code

Create a new folder within tests and copy the main.py file. Then create the input.ini file for your project. 
