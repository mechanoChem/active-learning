#NiAl 

from tensorflow import keras
import sys, os

import numpy as np
import json
from pathlib import Path


print('recreate model...')
dirpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirpath)

# from active_learning.model.idnn_model import IDNN_transforms
from active_learning.model.idnn import IDNN 
# from active_learning.data_collector.TransformsModule import transforms

casm_version = sys.argv[5]


print('read input...')
# Read in the casm Monte Carlo input file
input_file = sys.argv[1]
with open(input_file) as fin:
    inputs = json.load(fin)

# print(sys.argv)

# directory = os.path.abspath('../../../../LCO_row')
# sys.path.insert(0,directory)
from active_learning.workflow.dictionary import Dictionary

# directory='/Users/jamieholber/Desktop/Software/active-learning/tests/LCO_zigzag/'
directory=sys.argv[7]

input_path = directory+ '/input_predicting.ini'
dictionary = Dictionary(input_path)
[rnd, step] = dictionary.get_category_values('Restart')
# rnd= dictionary.


from active_learning.model.idnn_model import IDNN_Model 
model = IDNN_Model(dictionary)
model.load_trained_model(rnd)

phi = []
kappa  = []
T = []
[relevent_indices] = dictionary.get_individual_keys('CASM_Surrogate',['relevent_indices'])

if casm_version== 'NiAl':
    for comp in inputs["driver"]["conditions_list"]:
        T = comp["temperature"]
        phitemp = []
        kappatemp = []
        for i in range(dim):
            phitemp.append(comp["phi"][i][0])
            kappatemp.append(comp["kappa"][i][0])
        phi.append(phitemp)
        kappa.append(kappatemp)
if casm_version=='LCO':
    for comp in inputs["driver"]["custom_conditions"]:
         T.append(comp["temperature"])
         phitemp = []
         kappatemp = []
         for i in range(dim):
            phitemp.append(comp["bias_phi"]["{}".format(i)])
            kappatemp.append(comp["bias_kappa"]["{}".format(i)])
         phi.append(phitemp)
         kappa.append(kappatemp)
if casm_version=='row':
    for comp in inputs["driver"]["custom_conditions"]:
         T.append(comp["temperature"])
         phitemp = []
         kappatemp = []
         phiscomp = comp["order_parameter_quad_pot_vector"]
         kappascomp = comp["order_parameter_quad_pot_target"]
        #  phitemp = phiscomp[rows_to_keep]
         for j in relevent_indices:
                # print(phiscomp[j])
                # print('j',j)
                # print(kappascomp[j])
                phitemp.append(phiscomp[j])
                kappatemp.append(kappascomp[j])
         phi.append(phitemp)
         kappa.append(kappatemp)


phi = np.array(phi)*32*32
eta = np.array(kappa )/32 # Since it's just for testing the workflow, we'll take eta as kappa
T = np.array(T)
T = np.reshape(T,(len(T),1))

# print('eta',eta[0:5,:])
# print('eta',kappa[0:5,:])

print('predicting...')





# data = np.loadtxt('/Users/jamieholber/Desktop/Software/active-learning/Output/prediction_rnd0.txt')

# eta = data[:,0:4]
# mu =data[:,4:8]
# print('mu og',mu)
# T = np.ones((np.shape(eta)[0],1))*260

# print('eta',eta)

pred = model.predict([eta,T])

# print('pred',pred)

mu = pred[1]
# print("mu_predicted",mu)
eta = eta
phi = phi
kappa = eta #+ 0.5*mu/phi # Back out what kappa would have been

#keras.backend.clear_session()

print('write output...')
# Write out a limited CASM-like results file
results = {"T": T.tolist()}
for i in range(len(relevent_indices)):
    if casm_version=='NiAl':
        results["kappa_{}".format(i)] = kappa[:,i].tolist()
        results["phi_{}".format(i)] = phi[:,i].tolist()
        results["<op_val({})>".format(i)] = eta[:,i].tolist()
    elif casm_version=='LCO':
        results[f"Bias_kappa({i})"] = kappa[:,i].tolist()
        results[f"Bias_phi({i})"] = phi[:,i].tolist()
        results[f"<order_param({i})>"] = eta[:,i].tolist()
    elif casm_version=='row':
        results[f"Bias_kappa({i})"] = kappa[:,i].tolist()
        results[f"Bias_phi({i})"] = phi[:,i].tolist()
        results[f"<order_param({i})>"] = eta[:,i].tolist()
        results[f"<mu({i})>"] = mu[:,i].tolist()
        # results[f"<msu({i})>"] = eta[:,i].tolist()


with open('results.json','w') as fout:
    json.dump(results,fout,sort_keys=True, indent=4)

