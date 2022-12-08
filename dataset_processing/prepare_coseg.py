#!/usr/bin/env python3

import bpy
import sys
from tqdm import tqdm
from subprocess import Popen, PIPE
import argparse
from pathlib import Path
import importlib
import io
from contextlib import redirect_stdout

def import_parents(level=1):
    global __package__
    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[level]

    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError:
        # already removed
        pass

    __package__ = '.'.join(parent.parts[len(top.parts):])
    importlib.import_module(__package__)

if __name__ == '__main__' and __package__ is None:
    import_parents(level=1)

from common.bpy_util import normalize_scale


def main():
    if '--' in sys.argv:
        # refer to https://b3d.interplanety.org/en/how-to-pass-command-line-arguments-to-a-blender-python-script-or-add-on/
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        raise Exception("Expected \'--\' followed by arguments to the script")

    parser = argparse.ArgumentParser("prepare_coseg")
    parser.add_argument('--shapes-dir', type=str, required=True, help='Path to COSEG raw shapes directory which contains the .off files')
    parser.add_argument('--target-dataset-dir', type=str, required=True, help='Path to dataset directory where the normalized COSEG. obj files will be stored')
    parser.add_argument('--target-phase', type=str, required=True, help='The name of the phase will hold the .obj files, e.g. \"coseg\"')
    args = parser.parse_args(argv)

    shapes_dir = Path(args.shapes_dir)
    target_dataset_dir = Path(args.target_dataset_dir).expanduser()
    target_phase_dir = target_dataset_dir.joinpath(args.target_phase)
    target_phase_dir.mkdir(exist_ok=True)
    target_obj_gt_dir = target_phase_dir.joinpath('obj_gt')
    target_obj_gt_dir.mkdir(exist_ok=True)

    print("Converting .off files to obj files...")
    for off_file in tqdm(list(shapes_dir.glob('*.off'))):
        obj_file = target_obj_gt_dir.joinpath(f'{off_file.stem}.obj')
        if obj_file.is_file():
            continue
        path_to_converter = Path(__file__).parent.joinpath('model-converter-python', 'convert.py').resolve()
        cmd = [str(path_to_converter), '-i', str(off_file), '-o', str(obj_file)]
        print(" ".join(cmd))
        process = Popen(cmd, stdout=PIPE)
        process.wait()

    print("Normalizing obj files...")
    for obj_file in tqdm(list(target_obj_gt_dir.glob("*.obj"))):
        with redirect_stdout(io.StringIO()):
            bpy.ops.import_scene.obj(filepath=str(obj_file))
            imported_object = bpy.context.selected_objects[0]
            imported_object.data.materials.clear()
            normalize_scale(imported_object)
            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = imported_object
            imported_object.select_set(True)
            bpy.ops.export_scene.obj(filepath=str(obj_file), use_selection=True, use_materials=False, use_triangles=True)
            bpy.ops.object.delete()


if __name__ == "__main__":
    main()
