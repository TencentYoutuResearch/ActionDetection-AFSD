import numpy as np

class_name_path = 'anet_annotations/action_name.txt'
classes = np.loadtxt(class_name_path, np.str, delimiter='\n')

class_to_id = {}
id_to_class = {}

for i, label in enumerate(classes):
    class_to_id[label] = i + 1
    id_to_class[i + 1] = label