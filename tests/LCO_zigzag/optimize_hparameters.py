import sys
import numpy as np
rnd = 1
i = 1
sys.path.append('/Users/jamieholber/Desktop/Software/AL_from_expanse/active-learning')
from active_learning.model.idnn_model import IDNN_Model
from active_learning.workflow.dictionary import Dictionary
dict = Dictionary("../../Output/sim1/input.ini","/Users/jamieholber/Desktop/Software/AL_from_expanse/active-learning/tests/LCO_zigzag")
model  = IDNN_Model(dict)
valid_loss,params = model.train_rand_idnn(rnd,i)
if not np.isnan(valid_loss):
	fout = open('../../Output/sim1/hparameters_1.txt','w')
	fout.write('hparameters += [[{},"{}_{}",{}]]'.format(params,rnd,i,valid_loss))
	fout.close()
