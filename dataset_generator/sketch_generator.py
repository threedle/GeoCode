#!/usr/bin/env python3

import sys
import traceback
import bpy
import time
from mathutils import Vector
import math
import mathutils
import random
import argparse
from tqdm import tqdm
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

from common.bpy_util import normalize_scale, look_at, del_obj, clean_scene, use_gpu_if_available
from common.file_util import get_recipe_yml_obj, hash_file_name


"""
Shader references:
    pencil shader - https://www.youtube.com/watch?v=71KGlu_Yxtg
    white background (compositing) - https://www.youtube.com/watch?v=aegiN7XeLow
    creating transparent object - https://www.katsbits.com/codex/transparency-cycles/
"""


def main(dataset_dir: Path, phase, parallel, mod):
    try:
        clean_scene()

        use_gpu_if_available()  # also switches to Cycles

        # setup to avoid rendering surfaces and only render the freestyle curves
        bpy.context.view_layer.use_pass_z = False
        bpy.context.view_layer.use_pass_combined = False
        bpy.context.view_layer.use_sky = False
        bpy.context.view_layer.use_solid = False
        bpy.context.view_layer.use_volumes = False
        bpy.context.view_layer.use_strand = True  # freestyle curves
        bpy.context.scene.render.use_freestyle = True
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGB'
        bpy.context.scene.view_settings.view_transform = 'Standard'

        if "Along Stroke" not in bpy.data.linestyles['LineStyle'].thickness_modifiers:
            bpy.ops.scene.freestyle_thickness_modifier_add(type='ALONG_STROKE')
            bpy.data.linestyles["LineStyle"].thickness_modifiers["Along Stroke"].mapping = 'CURVE'
            bpy.data.linestyles["LineStyle"].thickness_modifiers["Along Stroke"].curve.curves[0].points[0].location = (0.0, 0.44375)
            bpy.data.linestyles["LineStyle"].thickness_modifiers["Along Stroke"].curve.curves[0].points.new(0.031879, 0.6875)
            bpy.data.linestyles["LineStyle"].thickness_modifiers["Along Stroke"].curve.curves[0].points.new(0.088926, 0.8625)
            bpy.data.linestyles["LineStyle"].thickness_modifiers["Along Stroke"].curve.curves[0].points.new(0.104027, 0.918751)
            bpy.data.linestyles["LineStyle"].thickness_modifiers["Along Stroke"].curve.curves[0].points.new(0.213087, 0.5875)
            bpy.data.linestyles["LineStyle"].thickness_modifiers["Along Stroke"].curve.curves[0].points.new(0.315436, 0.887501)
            bpy.data.linestyles["LineStyle"].thickness_modifiers["Along Stroke"].curve.curves[0].points.new(0.404362, 0.64375)
            bpy.data.linestyles["LineStyle"].thickness_modifiers["Along Stroke"].curve.curves[0].points.new(0.463088, 0.55625)
            bpy.data.linestyles["LineStyle"].thickness_modifiers["Along Stroke"].curve.curves[0].points.new(0.520134, 0.7125)
            bpy.data.linestyles["LineStyle"].thickness_modifiers["Along Stroke"].curve.curves[0].points.new(0.545302, 0.975001)
            bpy.data.linestyles["LineStyle"].thickness_modifiers["Along Stroke"].curve.curves[0].points.new(0.630872, 0.7)
            bpy.data.linestyles["LineStyle"].thickness_modifiers["Along Stroke"].curve.curves[0].points.new(0.778523, 0.76875)
            bpy.data.linestyles["LineStyle"].thickness_modifiers["Along Stroke"].curve.curves[0].points.new(0.892618, 0.55)
            bpy.data.linestyles["LineStyle"].thickness_modifiers["Along Stroke"].curve.curves[0].points[-1].location = (1.0, 0.70625)

        # compositing
        if not bpy.context.scene.use_nodes:
            bpy.context.scene.use_nodes = True
            render_layers_node = bpy.context.scene.node_tree.nodes['Render Layers']
            composite_node = bpy.context.scene.node_tree.nodes['Composite']
            alpha_over_node = bpy.context.scene.node_tree.nodes.new('CompositorNodeAlphaOver')
            bpy.context.scene.node_tree.links.new(render_layers_node.outputs['Image'], alpha_over_node.inputs[2])
            bpy.context.scene.node_tree.links.new(alpha_over_node.outputs['Image'], composite_node.inputs['Image'])

        recipe_file_path = dataset_dir.joinpath('recipe.yml')
        recipe_yml_obj = get_recipe_yml_obj(recipe_file_path)
        camera_angles = recipe_yml_obj['camera_angles_train'] + recipe_yml_obj['camera_angles_test']
        # euler setting
        radius = 2
        eulers = [mathutils.Euler((math.radians(camera_angle[0]), 0.0, math.radians(camera_angle[1])), 'XYZ') for camera_angle in camera_angles]

        obj_gt_dir = dataset_dir.joinpath(phase, 'obj_gt')
        path_to_sketches = dataset_dir.joinpath(phase, 'sketches')  # output folder
        if (parallel == 1 or mod == 0) and not path_to_sketches.is_dir():
            path_to_sketches.mkdir()

        if parallel == 1 and mod != 0:
            while not path_to_sketches.is_dir():
                time.sleep(2)

        obj_files = sorted(obj_gt_dir.glob('*.obj'))
        # filter out files that were already processed
        obj_files = [file for file in obj_files if
                     not all(
                         list(path_to_sketches.glob(f'{file.stem}_{camera_angle[0]}_{camera_angle[1]}.png'))
                         for camera_angle in camera_angles)]
        # remove any file that is not handled in this job
        if parallel > 1:
            obj_files = [file for file in obj_files if hash_file_name(file.name) % parallel == mod]

        for obj_file in tqdm(obj_files):
            file_name = obj_file.name

            filepath = obj_gt_dir.joinpath(file_name)
            bpy.ops.wm.obj_import(filepath=str(filepath), forward_axis='NEGATIVE_Z', up_axis='Y', filter_glob="*.obj;*.mtl")
            obj = bpy.context.selected_objects[0]

            # normalize the object
            normalize_scale(obj)

            for i, eul in enumerate(eulers):
                filename_no_ext = obj_file.stem
                target_file_name = f"{filename_no_ext}_{camera_angles[i][0]:.1f}_{camera_angles[i][1]:.1f}.png"
                target_file = path_to_sketches.joinpath(target_file_name)
                if target_file.is_file():
                    continue

                # camera setting
                cam_pos = mathutils.Vector((0.0, -radius, 0.0))
                cam_pos.rotate(eul)
                if i < 4:
                    # camera position perturbation
                    rand_x = random.uniform(-2.0, 2.0)
                    rand_z = random.uniform(-3.0, 3.0)
                    eul_perturb = mathutils.Euler((math.radians(rand_x), 0.0, math.radians(rand_z)), 'XYZ')
                    cam_pos.rotate(eul_perturb)

                scene = bpy.context.scene
                bpy.ops.object.camera_add(enter_editmode=False, location=cam_pos)
                new_camera = bpy.context.active_object
                new_camera.name = "camera_tmp"
                new_camera.data.name = "camera_tmp"
                new_camera.data.lens_unit = 'FOV'
                new_camera.data.angle = math.radians(60)
                look_at(new_camera, Vector((0.0, 0.0, 0.0)))

                # render
                scene.camera = new_camera
                scene.render.filepath = str(target_file)
                scene.render.resolution_x = 224
                scene.render.resolution_y = 224
                bpy.context.scene.cycles.samples = 10
                # debug
                if False:
                    debug_file_path = dataset_dir / "gen_sketch_debug.blend"
                    bpy.ops.wm.save_as_mainfile(filepath=str(debug_file_path))
                    return
                bpy.ops.render.render(write_still=True)

                # prepare for the next camera
                del_obj(new_camera)

            # delete the obj to prepare for the next one
            del_obj(obj)

        # clean the scene
        clean_scene()
    except Exception as e:
        print(repr(e))
        print(traceback.format_exc())


if __name__ == "__main__":
    argv = sys.argv
    if '--' in sys.argv:
        # refer to https://b3d.interplanety.org/en/how-to-pass-command-line-arguments-to-a-blender-python-script-or-add-on/
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        raise Exception("Expected \'--\' followed by arguments to the script")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--parallel', type=int, default=1, help='Number of processes that will run the script')
    parser.add_argument('--mod', type=int, default=0, help='The modulo for this process to match files\' hash')
    parser.add_argument('--phases', type=str, required=True, nargs='+', help='List of phases to generate the sketches for')

    args = parser.parse_args(argv)

    # hide the main collections (if it is already hidden, there is no effect)
    bpy.context.view_layer.layer_collection.children['Main'].hide_viewport = True
    bpy.context.view_layer.layer_collection.children['Main'].exclude = True

    dataset_dir = Path(args.dataset_dir).expanduser()
    phases = args.phases
    for phase in phases:
        main(dataset_dir, phase, args.parallel, args.mod)
