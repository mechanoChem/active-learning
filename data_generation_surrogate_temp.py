#NiAl 

from tensorflow import keras
import sys, os

import numpy as np
from idnn_old import IDNN
from transform_layer import Transform
import json


print('recreate model...')
#path = os.path.dirname(os.getcwd())
#sys.path.append(path)
#print(path)


hidden_layer = [int(p) for p in sys.argv[2].split(',')]
shapelist = [int(p) for p in sys.argv[3].split(',')]
dim = int(sys.argv[4])
casm_version = sys.argv[5]
activationinput =  sys.argv[6]
path = sys.argv[7]
sys.path.append(path)
from TransformsModule import transforms


hidden_layers = hidden_layer 
idnn = IDNN(dim,
            hidden_layers,
            activation = activationinput,
            transforms=transforms,
            final_bias=True)

idnn.build(input_shape=(shapelist[0],shapelist[1]))
for i in range(len(hidden_layers)+1):
    filew = open(path + '/surrogate_weights/weights_{}.txt'.format(i))
    linew=filew.readlines() 
    filew.close()
    w = []
    for j in range(len(linew)-1):
        linew[j+1]=linew[j+1].strip("\n")
        linew[j+1]=linew[j+1].strip("\b")
        w.append(linew[j+1].split(' '))
    w = np.array(w, dtype='float')
    fileb = open(path + '/surrogate_weights/bias_{}.txt'.format(i))
    lineb=fileb.readlines() 
    fileb.close()
    b = []
    for j in range(len(lineb)-1):
        lineb[j+1]=lineb[j+1].strip("\n")
        lineb[j+1]=lineb[j+1].strip("\b")
        b.append(lineb[j+1])
    b = np.array(b, dtype='float')
    idnn.dnn_layers[i].set_weights([w,b])

print('read input...')
# Read in the casm Monte Carlo input file
input_file = sys.argv[1]
with open(input_file) as fin:
    inputs = json.load(fin)

phi = []
kappa  = []
T = []
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


phi = np.array(phi)
eta = np.array(kappa) # Since it's just for testing the workflow, we'll take eta as kappa
T = np.array(T)

print('predicting...')
pred = idnn.predict(eta)
mu = pred[1]
for i in range(len(T)):
    temp = T[i]
    for j in range(7):
        mu[i,j] = mu[i,j] + (2.772*pow(10,-10)*pow(temp,3) - 1.551*pow(10,-7)*pow(temp,2) + temp*2.67*pow(10,-5) ) -  (2.772*pow(10,-10)*pow(300,3) - 1.551*pow(10,-7)*pow(300,2) + 2.67*pow(10,-5)*(300)) - 0.00048443 #adjust temp
#mu = mu +  (2.772*pow(10,-10)*pow(T,3) - 1.551*pow(10,-7)*pow(T,2) + T*2.67*pow(10,-5) ) -  (2.772*pow(10,-10)*pow(300,3) - 1.551*pow(10,-7)*pow(300,2) + 2.67*pow(10,-5)*(300)) #adjust temp 
kappa = eta + 0.5*mu/phi # Back out what kappa would have been

#keras.backend.clear_session()

print('write output...')
# Write out a limited CASM-like results file
results = {"T": T.tolist()}
for i in range(dim):
    if casm_version=='NiAl':
        results["kappa_{}".format(i)] = kappa[:,i].tolist()
        results["phi_{}".format(i)] = phi[:,i].tolist()
        results["<op_val({})>".format(i)] = eta[:,i].tolist()
    elif casm_version=='LCO':
        results[f"Bias_kappa({i})"] = kappa[:,i].tolist()
        results[f"Bias_phi({i})"] = phi[:,i].tolist()
        results[f"<order_param({i})>"] = eta[:,i].tolist()


with open('results.json','w') as fout:
    json.dump(results,fout,sort_keys=True, indent=4)
