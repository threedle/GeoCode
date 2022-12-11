import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from typing import List, Optional, Dict


arithmetic_symbols = ['and', 'or', 'not', '(', ')', '<', '<=' , '>', '>=', '==', '-', '+', '/', '*']

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


@dataclass
class ParamDescriptor:
    input_type: str
    num_classes: int
    step: float
    classes: np.ndarray
    normalized_classes: np.ndarray
    min_val: float
    max_val: float
    visibility_condition: str
    is_regression: bool
    normalized_acc_threshold: float

    def is_visible(self, param_values_map):
        assert param_values_map
        if not self.visibility_condition:
            return True
        is_visible_cond = " ".join([(word if (word in arithmetic_symbols or isfloat(word) or word.isnumeric()) else (f"param_values_map[\"{word}\"] == 1" if 'is_' in word else f"param_values_map[\"{word}\"]")) for word in self.visibility_condition.split(" ")])
        return eval(is_visible_cond)


class ParamDescriptors:
    def __init__(self, recipe_yml_obj, inputs_to_eval, use_regression=False, train_with_visibility_label=True):
        self.epsilon = 1e-6
        self.recipe_yml_obj = recipe_yml_obj
        self.inputs_to_eval = inputs_to_eval
        self.use_regression = use_regression
        self.train_with_visibility_label = train_with_visibility_label
        self.__overall_num_of_classes_without_visibility_label = 0
        self.param_descriptors_map: Optional[Dict[str, ParamDescriptor]] = None
        self.__constraints: Optional[List[str]] = None

    def check_constraints(self, param_values_map):
        assert param_values_map
        for constraint in self.get_constraints():
            is_fulfilled = " ".join([(word if (word in arithmetic_symbols or isfloat(word) or word.isnumeric()) else (f"param_values_map[\"{word}\"] == 1" if 'is_' in word else f"param_values_map[\"{word}\"]")) for word in constraint.split(" ")])
            if not eval(is_fulfilled):
                return False
        return True

    def get_constraints(self):
        if self.__constraints:
            return self.__constraints
        self.__constraints = []
        if 'constraints' in self.recipe_yml_obj:
            for constraint_name, constraint in self.recipe_yml_obj['constraints'].items():
                self.__constraints.append(constraint)
        return self.__constraints

    def get_param_descriptors_map(self):
        if self.param_descriptors_map:
            return self.param_descriptors_map
        recipe_yml_obj = self.recipe_yml_obj  # for readability
        param_descriptors_map = OrderedDict()
        visibility_conditions = {}
        if 'visibility_conditions' in recipe_yml_obj:
            visibility_conditions = recipe_yml_obj['visibility_conditions']

        for i, param_name in enumerate(self.inputs_to_eval):
            is_regression = False
            normalized_acc_threshold = None
            if " x" in param_name or " y" in param_name or " z" in param_name:
                input_type = recipe_yml_obj['data_types'][param_name[:-2]]['type']
            else:
                input_type = recipe_yml_obj['data_types'][param_name]['type']
            if input_type == 'Integer' or input_type == 'Boolean':
                max_val = recipe_yml_obj['dataset_generation'][param_name]['max']
                min_val = recipe_yml_obj['dataset_generation'][param_name]['min']
                step = 1
                num_classes = max_val - min_val + step
                self.__overall_num_of_classes_without_visibility_label += num_classes
                classes = np.arange(min_val, max_val + self.epsilon, step)
                normalized_classes = classes - min_val
                # visibility label adjustment
                if self.train_with_visibility_label:
                    for vis_cond_name, vis_cond in visibility_conditions.items():
                        if vis_cond_name in param_name:
                            num_classes += 1
                            classes = np.concatenate((np.array([-1.0]), classes))
                            normalized_classes = np.concatenate((np.array([-1.0]), normalized_classes))
                            break
            elif input_type == 'Float':
                max_val = recipe_yml_obj['dataset_generation'][param_name]['max']
                min_val = recipe_yml_obj['dataset_generation'][param_name]['min']
                samples = recipe_yml_obj['dataset_generation'][param_name]['samples']
                step, num_classes, classes, normalized_classes, is_regression, normalized_acc_threshold \
                    = self._handle_float(param_name, samples, min_val, max_val, visibility_conditions)
            elif input_type == 'Vector':
                axis = param_name[-1]
                param_name_no_axis = param_name[:-2]
                max_val = recipe_yml_obj['dataset_generation'][param_name_no_axis][axis]['max']
                min_val = recipe_yml_obj['dataset_generation'][param_name_no_axis][axis]['min']
                samples = recipe_yml_obj['dataset_generation'][param_name_no_axis][axis]['samples']
                step, num_classes, classes, normalized_classes, is_regression, normalized_acc_threshold \
                    = self._handle_float(param_name_no_axis, samples, min_val, max_val, visibility_conditions)
            else:
                raise Exception(f'Input type [{input_type}] is not supported yet')

            visibility_condition = None
            for vis_cond_name, vis_cond in visibility_conditions.items():
                if vis_cond_name in param_name:
                    visibility_condition = vis_cond
                    break
            param_descriptors_map[param_name] = ParamDescriptor(input_type, num_classes, step, classes,
                                                                normalized_classes, min_val, max_val,
                                                                visibility_condition, is_regression,
                                                                normalized_acc_threshold)
        self.param_descriptors_map = param_descriptors_map
        return self.param_descriptors_map

    def _handle_float(self, param_name, samples, min_val, max_val, visibility_conditions):
        """
        :param param_name: the parameter name, if the parameter is a vector, the axis should be omitted
        :param samples: the number of samples requested in the recipe file
        :param min_val: the min value allowed in the recipe file
        :param max_val: the max value allowed in the recipe file
        :param visibility_conditions: visibility conditions from the recipe file
        :return: step, num_classes, classes, normalized_classes, is_regression, normalized_acc_threshold
        """
        is_regression = False
        normalized_acc_threshold = None
        if not self.use_regression:
            step = (max_val - min_val) / (samples - 1)
            classes = np.arange(min_val, max_val + self.epsilon, step)
            classes = classes.astype(np.float64)
            normalized_classes = (classes - min_val) / (max_val - min_val)
            normalized_classes = normalized_classes.astype(np.float64)
            num_classes = classes.shape[0]
            self.__overall_num_of_classes_without_visibility_label += num_classes
            # visibility label adjustment
            if self.train_with_visibility_label:
                for vis_cond_name, vis_cond in visibility_conditions.items():
                    if vis_cond_name in param_name:
                        num_classes += 1
                        classes = np.concatenate((np.array([-1.0]), classes))
                        normalized_classes = np.concatenate((np.array([-1.0]), normalized_classes))
                        break
        else:
            step = 0
            num_classes = 2  # one for prediction and one for visibility label
            classes = None
            normalized_classes = None
            is_regression = True
            normalized_acc_threshold = 1 / (2 * (samples - 1))
        return step, num_classes, classes, normalized_classes, is_regression, normalized_acc_threshold


    def convert_prediction_vector_to_map(self, pred_vector, use_regression=False):
        """
        :param pred_vector: predicted vector from the network
        :param use_regression: whether we use regression for float values
        :return: map object representing the shape
        """
        pred_vector = pred_vector.squeeze()
        assert len(pred_vector.shape) == 1
        shape_map = {}
        idx = 0
        param_descriptors_map = self.get_param_descriptors_map()
        for param_name in self.inputs_to_eval:
            param_descriptor = param_descriptors_map[param_name]
            input_type = param_descriptor.input_type
            classes = param_descriptor.classes
            num_classes = param_descriptor.num_classes
            if input_type == 'Float' or input_type == 'Vector':
                if not use_regression:
                    normalized_pred_class = int(np.argmax(pred_vector[idx:idx + num_classes]))
                    pred_val = float(classes[normalized_pred_class])
                else:
                    min_val = param_descriptor.min_val
                    max_val = param_descriptor.max_val
                    pred_val = -1.0
                    if float(pred_vector[idx + 1]) < 0.5:  # visibility class
                        pred_val = (float(pred_vector[idx]) * (max_val - min_val)) + min_val
            else:
                # Integer or Boolean
                normalized_pred_class = int(np.argmax(pred_vector[idx:idx + num_classes]))
                pred_val = int(classes[normalized_pred_class])
            if input_type == 'Vector':
                if param_name[:-2] not in shape_map:
                    shape_map[param_name[:-2]] = {}
                shape_map[param_name[:-2]][param_name[-1]] = pred_val
            else:
                shape_map[param_name] = pred_val
            idx += num_classes
        return shape_map

    def get_overall_num_of_classes_without_visibility_label(self):
        self.get_param_descriptors_map()
        return self.__overall_num_of_classes_without_visibility_label

    def expand_target_vector(self, targets):
        """
        :param targets: 1-dim target vector which includes a single normalized value for each parameter
        :return: 1-dim vector where each parameter prediction is in one-hot representation
        """
        targets = targets.squeeze()
        assert len(targets.shape) == 1
        res_vector = np.array([])
        param_descriptors = self.get_param_descriptors_map()
        for i, param_name in enumerate(self.inputs_to_eval):
            param_descriptor = param_descriptors[param_name]
            num_classes = param_descriptor.num_classes
            if param_descriptor.is_regression:
                val = targets[i].reshape(1).item()
                if val == -1.0:
                    res_vector = np.concatenate((res_vector, np.array([0.0, 1.0])))
                else:
                    res_vector = np.concatenate((res_vector, np.array([val, 0.0])))
            else:
                normalized_classes = param_descriptor.normalized_classes
                normalized_gt_class_idx = int(np.where(abs(normalized_classes - targets[i].item()) < 1e-3)[0].item())
                one_hot = np.eye(num_classes)[normalized_gt_class_idx]
                res_vector = np.concatenate((res_vector, one_hot))
        return res_vector
