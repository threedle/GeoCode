#!/usr/bin/env python3

import sys
import bpy
import argparse
import traceback
from pathlib import Path
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

from common.file_util import get_recipe_yml_obj
from common.input_param_map import get_input_param_map, load_shape_from_yml
from common.bpy_util import clean_scene, select_shape, select_objs, get_geometric_nodes_modifier, save_obj


def save_obj_from_yml(args):
    if args.simplification_ratio:
        assert 0.0 <= args.simplification_ratio <= 1.0
    target_obj_file_path = Path(args.target_obj_file_path)
    assert target_obj_file_path.suffix == ".obj"
    clean_scene(start_with_strings=["Camera", "Light"])
    bpy.context.scene.view_layers['View Layer'].layer_collection.children['Main'].hide_viewport = False
    bpy.context.scene.view_layers['View Layer'].layer_collection.children['Main'].exclude = False
    obj = select_shape()
    gnodes_mod = get_geometric_nodes_modifier(obj)
    recipe_yml = get_recipe_yml_obj(args.recipe_file_path)
    input_params_map = get_input_param_map(gnodes_mod, recipe_yml)
    load_shape_from_yml(args.yml_file_path, input_params_map, ignore_sanity_check=args.ignore_sanity_check)
    dup_obj = save_obj(target_obj_file_path, simplification_ratio=args.simplification_ratio)
    bpy.data.collections["Main"].hide_render = False
    chair_obj = select_shape()
    dup_obj.hide_render = False
    chair_obj.hide_render = True
    dup_obj.data.materials.clear()
    select_objs(dup_obj)
    bpy.ops.object.delete()


def main():
    if '--' in sys.argv:
        # refer to https://b3d.interplanety.org/en/how-to-pass-command-line-arguments-to-a-blender-python-script-or-add-on/
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        raise Exception("Expected \'--\' followed by arguments to the script")

    parser = argparse.ArgumentParser(prog='save_obj')
    parser.add_argument('--recipe-file-path', type=str, required=True, help='Path to recipe.yml file')
    parser.add_argument('--yml-file-path', type=str, required=True, help='Path to yaml file to convert to object')
    parser.add_argument('--target-obj-file-path', type=str, required=True, help='Path the obj file that will be created')
    parser.add_argument('--simplification-ratio', type=float, default=None, help='Simplification ratio to decimate the mesh')
    parser.add_argument('--ignore-sanity-check', action='store_true', default=False, help='Do not check the shape\'s parameters')

    try:
        args = parser.parse_known_args(argv)[0]
        save_obj_from_yml(args)
    except Exception as e:
        print(repr(e))
        print(traceback.format_exc())

if __name__ == '__main__':
    main()
