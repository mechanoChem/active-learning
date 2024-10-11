#!/usr/bin/env python
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')

import sys
directory = os.path.abspath('../../')
sys.path.insert(0,directory)
from active_learning.workflow.workflow import Workflow


input_path = sys.argv[1]
workflow = Workflow(os.path.abspath(input_path))
workflow.main_workflow()