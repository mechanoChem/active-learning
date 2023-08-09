# active-learning

Active Learning Customizable Workflow. 

The workflow developed here can train a model, update model parameters, recommend data to sample, and collect the data.Currently the workflow includes an IDNN model, hyperpameter searches, monte carlo calculations, and data recommendations based on sobol sequence, billiardwalk, random space sampling, samping near wells, sampling near vertices, high error points, finding wells and sampling, and finding non-convexities and sampling. 

The workflow is readily adapatable for a variety of problems, and could be used for different models and different sampling mechanisms, including working with experimental results. 


To run:
python main.py input.ini


Requirements
matplotlib
tensorflow
sobol-seq
