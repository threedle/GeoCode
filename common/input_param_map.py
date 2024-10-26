import yaml
import random
import traceback
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from bpy.types import NodeInputs, Modifier
from common.bpy_util import select_shape, refresh_obj_in_viewport, get_geometric_nodes_modifier


@dataclass
class InputParam:
    gnodes_mod: Modifier
    input: NodeInputs
    axis: Optional[str]  # None indicates that this is not a vector
    possible_values: List

    def assign_random_value(self):
        self.assign_value(random.choice(self.possible_values))

    def assign_value(self, val):
        assert val in self.possible_values
        input_type = self.input.bl_label
        identifier = self.input.identifier
        if input_type == "Float":
            self.gnodes_mod[identifier] = val

        if input_type == "Integer":
            self.gnodes_mod[identifier] = int(val)

        if input_type == "Boolean":
            self.gnodes_mod[identifier] = int(val)

        if input_type == "Vector":
            axis_idx = ['x', 'y', 'z'].index(self.axis)
            self.gnodes_mod[identifier][axis_idx] = val

    def get_value(self):
        identifier = self.input.identifier
        if self.axis:
            axis_idx = ['x', 'y', 'z'].index(self.axis)
            return self.gnodes_mod[identifier][axis_idx]
        return self.gnodes_mod[identifier]

    def get_name_for_file(self):
        res = str(self.input.name) + ("" if not self.axis else "_" + self.axis)
        return res.replace(" ", "_")


def get_input_values(input, yml_gen_rule):
    min_value = None
    max_value = None
    if input.bl_label != 'Boolean':
        min_value = input.min_value
        max_value = input.max_value
    # override min and max with requested values from recipe yml file
    if 'min' in yml_gen_rule:
        requested_min_value = yml_gen_rule['min']
        if min_value and requested_min_value < min_value:
            if abs(min_value - requested_min_value) > 1e-6:
                raise Exception(
                    f'Requested a min value of [{requested_min_value}] for parameter [{input.name}], but min allowed is [{min_value}]')
            # otherwise min_value should remain input.min_value
        else:
            min_value = requested_min_value
    if 'max' in yml_gen_rule:
        requested_max_value = yml_gen_rule['max']
        if max_value and requested_max_value > max_value:
            if abs(max_value - requested_max_value) > 1e-6:
                raise Exception(
                    f'Requested a max value of [{requested_max_value}] for parameter [{input.name}], but max allowed is [{max_value}]')
            # otherwise max_value should remain input.max_value
        max_value = requested_max_value
    step = 1 if 'samples' not in yml_gen_rule else calculate_step(min_value, max_value, yml_gen_rule['samples'])
    res = np.arange(min_value, max_value + 1e-6, step)

    # convert to integers if needed
    if input.bl_label in ['Boolean', 'Integer']:
        res = list(res.astype(int))
    else:
        res = [round(x, 4) for x in list(res)]

    return res


def calculate_step(min_value, max_value, samples):
    return (max_value - min_value) / (samples - 1)


def get_input_param_map(gnodes_mod, yml):
    input_params_map = {}
    # loops through all the inputs in the geometric node group
    group_input_nodes = [node for node in gnodes_mod.node_group.nodes if node.type == 'GROUP_INPUT']
    assert len(group_input_nodes) > 0
    group_input_node = group_input_nodes[0]
    param_names = [param_name for param_name in group_input_node.outputs if len(param_name) > 0]
    for param_name in yml['dataset_generation']:
        if param_name not in param_names:
            raise Exception(f"Parameter named [{param_name}] was not found in geometry nodes input group.")
    for input in group_input_node.outputs:
        param_name = str(input.name)
        if len(param_name) == 0:
            continue
        # we only change inputs that are explicitly noted in the yaml object
        if param_name in yml['dataset_generation']:
            param_gen_rule = yml['dataset_generation'][param_name]
            if 'x' in param_gen_rule or 'y' in param_gen_rule or 'z' in param_gen_rule:
                # vector handling
                for idx, axis in enumerate(['x', 'y', 'z']):
                    if not axis in param_gen_rule:
                        continue
                    curr_param_values = get_input_values(input, param_gen_rule[axis])
                    input_params_map[f"{param_name} {axis}"] = InputParam(gnodes_mod, input, axis, curr_param_values)
            else:
                curr_param_values = get_input_values(input, param_gen_rule)
                input_params_map[param_name] = InputParam(gnodes_mod, input, None, curr_param_values)
    return input_params_map


def yml_to_shape(shape_yml_obj, input_params_map, ignore_sanity_check=False):
    try:
        # select the object in blender
        obj = select_shape()
        # get the geometric nodes modifier fo the object
        gnodes_mod = get_geometric_nodes_modifier(obj)

        # loops through all the inputs in the geometric node group
        group_input_nodes = [node for node in gnodes_mod.node_group.nodes if node.type == 'GROUP_INPUT']
        assert len(group_input_nodes) > 0
        group_input_node = group_input_nodes[0]
        for input in group_input_node.outputs:
            param_name = str(input.name)
            if len(param_name) == 0:
                continue
            if param_name not in shape_yml_obj:
                continue
            param_val = shape_yml_obj[param_name]
            if hasattr(param_val, '__iter__'):
                # vector handling
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    val = param_val[axis]
                    val = round(val, 4)
                    param_name_with_axis = f'{param_name} {axis}'
                    gnodes_mod[input.identifier][axis_idx] = val if abs(val + 1.0) > 0.1 else input_params_map[param_name_with_axis].possible_values[0].item()
                    assert gnodes_mod[input.identifier][axis_idx] >= 0.0
            else:
                param_val = round(param_val, 4)
                if not ignore_sanity_check:
                    err_msg = f'param_name [{param_name}] param_val [{param_val}] possible_values {input_params_map[param_name].possible_values}'
                    assert param_val == -1 or (param_val in input_params_map[param_name].possible_values), err_msg
                gnodes_mod[input.identifier] = param_val if (abs(param_val + 1.0) > 0.1) else (input_params_map[param_name].possible_values[0].item())
                # we assume that all input values are non-negative
                assert gnodes_mod[input.identifier] >= 0.0

        refresh_obj_in_viewport(obj)
    except Exception as e:
        print(repr(e))
        print(traceback.format_exc())


def load_shape_from_yml(yml_file_path, input_params_map, ignore_sanity_check=False):
    with open(yml_file_path, 'r') as f:
        yml_obj = yaml.load(f, Loader=yaml.FullLoader)
    yml_to_shape(yml_obj, input_params_map, ignore_sanity_check=ignore_sanity_check)


def load_base_shape_from_yml(recipe_file_path, input_params_map):
    print(f'Loading the base shape from the YML file [{recipe_file_path}]')

    with open(recipe_file_path, 'r') as f:
        yml_obj = yaml.load(f, Loader=yaml.FullLoader)

    yml_to_shape(yml_obj['base'], input_params_map)


def randomize_all_params(input_params_map):
    param_values_map = {}
    for param_name, input_param in input_params_map.items():
        param_values_map[param_name] = random.choice(input_param.possible_values)
    return param_values_map
