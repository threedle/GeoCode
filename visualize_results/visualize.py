#!/usr/bin/env python3

import sys
import bpy
import math
import random
import argparse
import traceback
import mathutils
from pathlib import Path
from mathutils import Vector
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

from common.file_util import hash_file_name
from common.bpy_util import clean_scene, setup_lights, select_objs, normalize_scale, del_obj, look_at


def add_3d_text(obj_to_align_with, text):
    """
    Adds 3D text object in front of the normalized object
    """
    bpy.ops.object.select_all(action='DESELECT')
    font_curve = bpy.data.curves.new(type="FONT", name="Font Curve")
    font_curve.body = text
    font_obj = bpy.data.objects.new(name="Font Object", object_data=font_curve)
    bpy.context.scene.collection.objects.link(font_obj)
    font_obj.select_set(True)
    bpy.context.view_layer.objects.active = font_obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    font_obj.location.x = obj_to_align_with.location.x
    font_obj.location.y = obj_to_align_with.location.y
    font_obj.location.z = 0
    font_obj.scale.x *= 0.2
    font_obj.scale.y *= 0.2
    font_obj.location.y -= 1
    return font_obj


def visualize_results(args):
    """
    Before using this method, a test dataset should be evaluated using the model
    """
    test_ds_dir = Path(args.dataset_dir, args.phase).expanduser()
    if not test_ds_dir.is_dir():
        raise Exception(f"Expected a \'{args.phase}\' dataset directory with 3D object to evaluate")

    results_dir = test_ds_dir.joinpath(f'results_{args.exp_name}')
    model_predictions_pc_dir = results_dir.joinpath('yml_predictions_pc')
    if not model_predictions_pc_dir.is_dir():
        raise Exception(f"Expected a \'results_{args.exp_name}/yml_predictions_pc\' directory with predictions from point clouds")

    model_predictions_sketch_dir = results_dir.joinpath('yml_predictions_sketch')
    if not model_predictions_sketch_dir.is_dir():
        raise Exception(f"Expected a \'results_{args.exp_name}/yml_predictions_sketch\' directory with predictions from skeches")

    obj_gt_dir = results_dir.joinpath('obj_gt')
    render_gt_dir = results_dir.joinpath('render_gt')
    obj_predictions_pc_dir = results_dir.joinpath('obj_predictions_pc')
    render_predictions_pc_dir = results_dir.joinpath('render_predictions_pc')
    obj_predictions_sketch_dir = results_dir.joinpath('obj_predictions_sketch')
    render_predictions_sketch_dir = results_dir.joinpath('render_predictions_sketch')

    work = [
        (obj_gt_dir, render_gt_dir, "GT"),  # render original 3D objs
        (obj_predictions_pc_dir, render_predictions_pc_dir, "PRED FROM PC"),  # render predictions from point cloud input
        (obj_predictions_sketch_dir, render_predictions_sketch_dir, "PRED FROM SKETCH")  # render predictions from sketch input
    ]

    try:
        clean_scene(start_with_strings=["Camera", "Light"])
        setup_lights()
        # hide the main collections
        bpy.context.scene.view_layers['View Layer'].layer_collection.children['Main'].hide_viewport = True
        bpy.context.scene.view_layers['View Layer'].layer_collection.children['Main'].exclude = True
        for obj_dir, render_dir, title in work:
            file_names = sorted([f.stem for f in obj_dir.glob("*.obj")])
            if args.parallel > 1:
                file_names = [file for file in file_names if hash_file_name(file) % args.parallel == args.mod]
            for file_name in file_names:
                original_obj_file_path = obj_dir.joinpath(f'{file_name}.obj')
                bpy.ops.import_scene.obj(filepath=str(original_obj_file_path))
                imported_object = bpy.context.selected_objects[0]
                imported_object.hide_render = False
                imported_object.data.materials.clear()
                normalize_scale(imported_object)
                title_obj = add_3d_text(imported_object, title)
                render_images(render_dir, file_name)
                select_objs(title_obj, imported_object)
                bpy.ops.object.delete()
    except Exception as e:
        print(repr(e))
        print(traceback.format_exc())


def render_images(target_dir: Path, file_name, suffix=None):
    # euler setting
    camera_angles = [
        [-30.0, -35.0]
    ]
    radius = 2
    eulers = [mathutils.Euler((math.radians(camera_angle[0]), 0.0, math.radians(camera_angle[1])), 'XYZ') for
              camera_angle in camera_angles]

    for i, eul in enumerate(eulers):
        target_file_name = f"{file_name}{(f'_{suffix}' if suffix else '')}_at_{camera_angles[i][0]:.1f}_{camera_angles[i][1]:.1f}.png"
        target_file = target_dir.joinpath(target_file_name)

        # camera setting
        cam_pos = mathutils.Vector((0.0, -radius, 0.0))
        cam_pos.rotate(eul)
        if i < 4:
            rand_x = random.uniform(-2.0, 2.0)
            rand_z = random.uniform(-5.0, 5.0)
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
        bpy.context.scene.cycles.samples = 5
        # disable the sketch shader
        bpy.context.scene.render.use_freestyle = False
        bpy.ops.render.render(write_still=True)

        # prepare for the next camera
        del_obj(new_camera)


def main():
    if '--' in sys.argv:
        # refer to https://b3d.interplanety.org/en/how-to-pass-command-line-arguments-to-a-blender-python-script-or-add-on/
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        raise Exception("Expected \'--\' followed by arguments to the script")

    parser = argparse.ArgumentParser(prog='dataset_generator')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--phase', type=str, required=True, help='E.g. train, test or val')
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--mod', type=int, default=None)

    try:
        args = parser.parse_known_args(argv)[0]
        visualize_results(args)
    except Exception as e:
        print(repr(e))
        print(traceback.format_exc())


if __name__ == '__main__':
    main()
