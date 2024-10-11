#!/usr/bin/env python
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')

import sys
directory = os.path.abspath('../../../al1/')
sys.path.insert(0,directory)
from active_learning.workflow.workflow import Workflow


input_path = sys.argv[1]
rnd = sys.argv[2]
workflow = Workflow(os.path.abspath(input_path),only_initialize=True,originalpath= os.path.dirname(__file__))


workflow.finalize(int(rnd))