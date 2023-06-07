#!/usr/bin/env python
import sys, os
import tensorflow as tf

import sys
# sys.path.append('../../active_learning/workflow/workfow')
# from workflow import Workflow
from active_learning.workflow.workflow import Workflow

# from temp import addNumbers

# list = [2, 3]
# addNumbers(*list)


# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass

input_path = sys.argv[1]
workflow = Workflow(input_path)
workflow.main_workflow()