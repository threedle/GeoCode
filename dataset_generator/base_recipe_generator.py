from __future__ import annotations  # to allow deferred annotations

import bpy
import sys
import argparse
from pathlib import Path
from typing import List, Set
from dataclasses import dataclass, field

import pip
pip.main(['install', 'sympy'])
pip.main(['install', 'pyyaml'])

from sympy import symbols, simplify_logic, And, Or, Eq, Ne, Not
from sympy.parsing.sympy_parser import parse_expr
import yaml

import importlib

def import_parents(level=1):
    global __package__
    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[level]

    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError:
        pass

    __package__ = '.'.join(parent.parts[len(top.parts):])
    importlib.import_module(__package__)

if __name__ == '__main__' and __package__ is None:
    import_parents(level=1)

from common.bpy_util import select_shape, get_geometric_nodes_modifier
from common.file_util import save_yml
from dataset_generator.dataset_generator import update_base_shape_in_yml, update_recipe_yml_obj_with_metadata


def OrAll(expressions):
    cond = False
    for expression in expressions:
        cond |= expression
    return cond
    

@dataclass
class TreeNode:
    name: str
    condition: False
    param_names_true: Set[str] = field(default_factory=set)
    param_names_false: Set[str] = field(default_factory=set)
    node_true: List[TreeNode] = field(default_factory=list)
    node_false: List[TreeNode] = field(default_factory=list)
    
    def tostring(self, is_true):
        print(f"is_true [{is_true}] name [{self.name}] condition [{self.condition}] params true {self.param_names_true} params false {self.param_names_false}")
        for node in self.node_true:
            node.tostring(True)
        for node in self.node_false:
            node.tostring(False)
            
    def get_all_param_names(self):
        def get_all_param_names_rec(node: TreeNode, param_names):
            param_names.update(node.param_names_true)
            param_names.update(node.param_names_false)
            for n in node.node_true:
                get_all_param_names_rec(n, param_names)
            for n in node.node_false:
                get_all_param_names_rec(n, param_names)
        param_names = set([])
        get_all_param_names_rec(self, param_names)
        return param_names
    
    def induce_condition(self, param_name):
        def induce_cond_rec(node: TreeNode, param_name: str):
            if param_name in node.param_names_true and param_name in node.param_names_false:
                return True
            elif node.condition == 'True' and param_name in node.param_names_true:
                return True
            elif node.condition == 'False' and param_name in node.param_names_false:
                return True
            elif node.condition == 'True' and param_name not in node.param_names_true:
                expressions = [(induce_cond_rec(n, param_name)) for n in node.node_true]
                cond = OrAll(expressions)
                return (cond)
            elif node.condition == 'False' and param_name not in node.param_names_false:
                expressions = [(induce_cond_rec(n, param_name)) for n in node.node_false]
                cond = OrAll(expressions)
                return (cond)
            elif param_name in node.param_names_true:
                expressions = [(induce_cond_rec(n, param_name)) for n in node.node_false]
                if not expressions:
                    return (node.condition)
                cond = OrAll(expressions)
                return ((node.condition) | (Not(node.condition) & (cond)))
            elif param_name in node.param_names_false:
                expressions = [(induce_cond_rec(n, param_name)) for n in node.node_true]
                if not expressions:
                    return Not(node.condition)
                cond = OrAll(expressions)
                return (Not(node.condition) | ((node.condition) & (cond)))
            else:
                expressions1 = [(induce_cond_rec(n, param_name)) for n in node.node_false]
                expressions2 = [(induce_cond_rec(n, param_name)) for n in node.node_true]
                cond1 = OrAll(expressions1)
                cond2 = OrAll(expressions2)
                if not expressions1 and not expressions2:
                    return False
                elif not expressions1:
                    return ((node.condition) & (cond2))
                else:
                    try:
                        return (Not(node.condition) & (cond1))
                    except:
                        print(f"[{node.condition}] [{cond1}]")
                return ((Not(node.condition) & (cond1)) | ((node.condition) & (cond2)))
        return induce_cond_rec(self, param_name)
    
    
def apply_operation(operation: str, term1, term2):
    if operation == 'LESS_THAN':
        return term1 < term2
    elif operation == 'LESS_THAN_OR_EQUAL':
        return term1 <= term2
    elif operation == 'GREATER_THAN':
        return term1 > term2
    elif operation == 'GREATER_THAN_OR_EQUAL':
        return term1 >= term2
    elif operation == 'EQUAL':
        return Eq(term1, term2)
    elif operation == 'NOT_EQUAL':
        return Ne(term1, term2)
    else:
        raise Exception(f"operation [{operation}] is not recognized")


def assemble_condition(link, symbols_map):
    def assemble_condition_rec(link, symbols_map):
        from_node = link.from_node
        if from_node.type == 'GROUP_INPUT':
            socket_name = link.from_socket.name
            if socket_name not in symbols_map:
                symbols_map[socket_name] = symbols(socket_name)
            if link.to_node.type == 'SWITCH':
                # booleans are represented as 0 or 1
                return Eq(symbols_map[socket_name], 1)
            return symbols_map[socket_name]
        elif from_node.type == 'COMPARE':
            data_type = from_node.data_type
            input_idx_offset = None
            if data_type == 'FLOAT':
                input_idx_offset = 0
            elif data_type == 'INT':
                input_idx_offset = 2
            else:
                raise Exception(f"data_type [{data_type}] is not supported")
            input1 = from_node.inputs[0 + input_idx_offset]
            input2 = from_node.inputs[1 + input_idx_offset]
            if input1.is_linked:
                term1 = assemble_condition_rec(input1.links[0], symbols_map)
            else:
                term1 = input1.default_value
            if input2.is_linked:
                term2 = assemble_condition_rec(input2.links[0], symbols_map)
            else:
                term2 = input2.default_value
            cond = apply_operation(from_node.operation, term1, term2)
            return cond
        elif from_node.type == 'SWITCH':
            raise Exception("Aggregated switches are not supported yet")
        else:
            raise Exception(f"No support for node_type of [{from_node.type}]")
    condition = assemble_condition_rec(link, symbols_map)
    return condition


def parse_switch(switch_node, symbols_map):
    switch_inputs = [input for input in switch_node.inputs if input.is_linked]
    switch_param_name = None
    link_true = None
    link_false = None
    condition = False  # if not connected, the input condition defaults to 'False'
    for switch_input in switch_inputs:
        if switch_input.name == 'Switch':
            assert switch_node.inputs[1].is_linked
            condition = assemble_condition(switch_input.links[0], symbols_map)
        elif switch_input.name == 'True':
            assert switch_node.inputs[15].is_linked
            link_true = switch_input.links[0]
        elif switch_input.name == 'False':
            assert switch_node.inputs[14].is_linked
            link_false = switch_input.links[0]
    return condition, link_true, link_false


def traverse_geo_node(link, tree_node: TreeNode, is_true, visited, symbols_map):
    from_node = link.from_node
    if from_node.type == 'SWITCH' and from_node.input_type == 'GEOMETRY':
        cond, link_true, link_false = parse_switch(from_node, symbols_map)
        print(f"{from_node.name} {link_true} {link_false}")
        tn = TreeNode(from_node.name, cond, set([]), set([]), [], [])
        if is_true:
            tree_node.node_true.append(tn)
        else:
            tree_node.node_false.append(tn)
        if link_true:
            traverse_geo_node(link_true, tn, True, visited, symbols_map)
        if link_false:
            traverse_geo_node(link_false, tn, False, visited, symbols_map)
        return
    if from_node.type == 'GROUP_INPUT':
        socket_name = link.from_socket.name
        if is_true:
            tree_node.param_names_true.add(socket_name)
        else:
            tree_node.param_names_false.add(socket_name)
        return
    for input in from_node.inputs:
        if not input.is_linked:
            continue
        for link in input.links:
            child_node = link.from_node
            vis_name = f"{child_node.name}_{link.from_socket.name}"
            #if vis_name not in visited:
            #    visited.add(vis_name)
            traverse_geo_node(link, tree_node, is_true, visited, symbols_map)
    

def assemble_rules(final_node_name):
    obj = bpy.data.objects['procedural shape']
    mod = obj.modifiers['GeometryNodes']
    ng = mod.node_group
    final_node = ng.nodes[final_node_name]
    final_link = final_node.inputs['Geometry'].links[0]
    
    # by default we assume the condition is False
    visited = set([])
    symbols_map = {}
    root = TreeNode("root", False, set([]), set([]), [], [])
    traverse_geo_node(final_link, root, False, visited, symbols_map)
    root.tostring(False)
    param_names = root.get_all_param_names()
#    param_names = set([k for k in symbols_map])
    #bool_param_names = [p for p in param_names if p.startswith("is_")]
    cond_map = {}
    
    # build the current parameter dict (used for a sanity check)
    param_value_map = {}
    group_input_nodes = [node for node in mod.node_group.nodes if node.type == 'GROUP_INPUT']
    assert len(group_input_nodes) > 0
    group_input_node = group_input_nodes[0]
    for input in group_input_node.outputs:
        param_name = str(input.name)
        if len(param_name) == 0:
            continue
    # for input in ng.interface.items_tree:
    # for param_name in [input.name for input in ng.interface.items_tree]:
        #print(param_name)
        # param_name = input.name
        #print(ng.inputs[param_name].identifier)
        print(input.identifier)
        value = mod[input.identifier]
        if input.bl_label == 'BOOLEAN':
            param_value_map[param_name] = 1 if value else 0
        else:
            param_value_map[param_name] = value
    print(param_value_map)
    
    print(f"Number of parameters [{len(param_names)}]")
    for param_name in param_names:
        raw_expression = root.induce_condition(param_name)
#        parsed_expression = parse_expr(raw_expression)
        simplified_expression = simplify_logic(raw_expression)
        cond_map[param_name] = str(simplified_expression)
        current_eval = parse_expr(cond_map[param_name], local_dict=param_value_map)
        print(f"{param_name}: {cond_map[param_name]} with current parameters evaluates to {current_eval}")
    # remove "True" conditions
    return {k: v for k, v in cond_map.items() if v != "True"}


def main(args):
    recipe_file_path = Path(args.recipe_file_path)
    obj = select_shape()
    # get the geometric nodes modifier for the object
    gnodes_mod = get_geometric_nodes_modifier(obj)
    recipe_yml_obj = update_base_shape_in_yml(gnodes_mod, recipe_file_path)
    update_recipe_yml_obj_with_metadata(recipe_yml_obj, gnodes_mod, write_dataset_generation=True)
    visibility_conditions = assemble_rules(args.final_node_name)
    if len(visibility_conditions) > 0:
        recipe_yml_obj['visibility_conditions'] = visibility_conditions
    else:
        if "visibility_conditions" in recipe_yml_obj:
            del recipe_yml_obj["visibility_conditions"]
    save_yml(recipe_yml_obj, recipe_file_path)



if __name__ == "__main__":
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        raise Exception("Expected \'--\' followed by arguments to the script")

    parser = argparse.ArgumentParser(prog='dataset_generator')
    parser.add_argument('--recipe-file-path', type=str, required=True, help='Path to the output recipe file.')
    parser.add_argument('--final-node-name', type=str, help="The name of the output node of the procedural shape, e.g. \"Realize Instances\".")

    args = parser.parse_args(argv)
    main(args)

"""
"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" "D:\TAU MSc\Semester 4\Thesis\Shape Editing\Procedural Chair Revisit\simple_ceiling_lamp_with_inputs.blend" -b --python dataset_generator/base_recipe_generator.py -- --recipe-file-path "D:\TAU MSc\Semester 4\Thesis\Shape Editing\Procedural Chair Revisit\recipe_ceiling_lamp.yml" --final-node-name "Realize Instances"
"""
