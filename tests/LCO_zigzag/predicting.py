import os,sys
import numpy as np

directory = os.path.abspath('../../../active-learning/')
sys.path.insert(0,directory)
from active_learning.workflow.dictionary import Dictionary

input_path = 'input_predicting.ini'
dictionary = Dictionary(input_path)
from active_learning.model.idnn_model import IDNN_Model 
model = IDNN_Model(dictionary)



for rnd in [0]:

    model.load_trained_model(rnd)

    data = '2d_slice_rnd21_0_1.txt'
    eta = np.genfromtxt(data,dtype=np.float32)[:,:7]
    mu = np.genfromtxt(data,dtype=np.float32)[:,7:14]
    print('mu_old',mu)
    T = np.ones((np.shape(eta)[0],1))*300
    eta_test = eta.copy()

    pred_new = model.predict([eta_test,T])

    free_new = pred_new[0]
    mu_new = pred_new[1]
    print(mu_new)
    # mu = pred[1]
    grad = pred_new[2]
    np.savetxt('surrogate_model/predicting_{}.txt'.format(rnd),np.hstack((eta,mu_new,free_new)))