#!/usr/bin/env python
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# CUDA_VISIBLE_DEVICES=""
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import sys
directory = os.path.abspath('../../../active-learning/')
sys.path.insert(0,directory)
# from workflow import Workflow
from active_learning.workflow.graph import only_graph


input_path = sys.argv[1]
only_graph(os.path.abspath(input_path))