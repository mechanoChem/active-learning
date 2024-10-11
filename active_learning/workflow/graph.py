from active_learning.model.idnn_model import IDNN_Model 
from active_learning.workflow.make_graph import graph
from active_learning.workflow.dictionary import Dictionary

def only_graph(input_path):
    input_path = input_path
    dict = Dictionary(input_path)
    model = IDNN_Model(dict)

    for i in range(1):
        graph(i, model,dict)