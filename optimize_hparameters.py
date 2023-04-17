import sys
import numpy as np
rnd = 250
i = 2
sys.path.append('/Users/jamieholber/mechanoChemMl_active_learning/examples/active_learning/idnn_hyperparameter_with_T')
from active_learning import Active_learning
hidden_units, learning_rate, valid_loss = Active_learning("LCO_test.ini").train_rand_idnn(rnd,i)
if not np.isnan(valid_loss):
	fout = open('hparameters_2.txt','w')
	fout.write('hparameters += [[{},{},"{}_{}",{}]]'.format(learning_rate,hidden_units,rnd,i,valid_loss))
	fout.close()
