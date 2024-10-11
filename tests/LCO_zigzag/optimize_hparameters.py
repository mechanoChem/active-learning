import sys
import numpy as np
rnd = 1
i = 1
sys.path.append('/Users/jamieholber/Desktop/Software/active-learning')
from active_learning.model.idnn_model import IDNN_Model
from active_learning.workflow.dictionary import Dictionary
dict = Dictionary("/Users/jamieholber/Desktop/Software/active-learning/tests/LCO_zigzag/input.ini")
model  = IDNN_Model(dict)
valid_loss,params = model.train_rand_idnn(rnd,i)
if not np.isnan(valid_loss):
	fout = open('../../Output_zigzag/300/sim4/hparameters_1.txt','w')
	fout.write('hparameters += [[{},"{}_{}",{}]]'.format(params,rnd,i,valid_loss))
	fout.close()
