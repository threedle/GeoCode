from enum import Enum
from typing import Dict
from common.param_descriptors import ParamDescriptor


class InputType(Enum):
    sketch = 'sketch'
    pc = 'pc'

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return self.value == other.value


def get_inputs_to_eval(recipe_yml_obj):
    inputs_to_eval = []
    for param_name, param_dict in recipe_yml_obj['dataset_generation'].items():
        is_vector = False
        for axis in ['x', 'y', 'z']:
            if axis in param_dict:
                inputs_to_eval.append(f'{param_name} {axis}')
                is_vector = True
        if not is_vector:
            inputs_to_eval.append(param_name)
    print("Inputs that will be evaluated:")
    print("\t" + "\n\t".join(inputs_to_eval))
    return inputs_to_eval


def calc_prediction_vector_size(param_descriptors_map: Dict[str, ParamDescriptor]):
    detailed_vec_size = [param_descriptor.num_classes for param_name, param_descriptor in param_descriptors_map.items()]
    print(f"Found [{len(detailed_vec_size)}] parameters with a combined number of classes of [{sum(detailed_vec_size)}]")
    return detailed_vec_size
