import os, sys

directory = os.path.abspath('../../../active-learning/')
sys.path.insert(0,directory)

from active_learning.workflow.workflow import Workflow


from active_learning.model.idnn import IDNN
from active_learning.model.idnn_model import IDNN_Model
from active_learning.workflow.dictionary import Dictionary
import numpy as np
# from CASM_wrapper import compileCASMOutput, loadCASMOutput
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import load_model
import pandas as pd 
from active_learning.data_recommended.DataRecommender import DataRecommender

rnd=1

dict = Dictionary('input.ini')
model = IDNN_Model(dict)
model.load_trained_model(rnd-1)
recommender = DataRecommender(model,dict)
recommender.construct_input_types()

recommender.get_latest_pred(rnd)
recommender.high_error(rnd)
# pred = model.predict([eta,T])
# free = pred[0]
# mu = pred[1]
# grad = pred[2]
# np.savetxt('model4.txt',np.hstack((eta,mu,free)))