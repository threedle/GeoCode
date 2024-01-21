#!/usr/bin/env python3

import bpy
import sys
from tqdm import tqdm
import argparse
from pathlib import Path
import importlib
import io
from contextlib import redirect_stdout
import multiprocessing

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

from common.bpy_util import normalize_scale


def normalize_obj(obj_file_path: Path):
    """
    saves a normalized version of the object in place (overwrites the file)
    this also takes care of joining separated parts (such as in ShapeNet dataset)
    """
    with redirect_stdout(io.StringIO()):
        bpy.ops.import_scene.obj(filepath=str(obj_file_path))
        imported_objects = bpy.context.selected_objects
        parent_obj = bpy.context.selected_objects[0]
        for obj in imported_objects:
            obj.data.materials.clear()
        bpy.context.view_layer.objects.active = parent_obj
        # join all the parts together
        if len(imported_objects) > 1:
            bpy.ops.object.join()
        normalize_scale(parent_obj)
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = parent_obj
        parent_obj.select_set(True)
        bpy.ops.export_scene.obj(filepath=str(obj_file_path), use_selection=True, use_materials=False, use_triangles=True)
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()



def main():
    """
    sequential processing of the files
    """
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        raise Exception("Expected \'--\' followed by arguments to the script")

    parser = argparse.ArgumentParser("prepare_coseg")
    parser.add_argument('--shapes-dir', type=str, required=True, help='Path to COSEG raw shapes directory which contains the .off files')
    args = parser.parse_args(argv)
    shapes_dir = Path(args.shapes_dir)
    print(f"Normalizing obj files in [{shapes_dir}]")
    for obj_file_path in tqdm(list(shapes_dir.glob("*.obj"))):
        normalize_obj(obj_file_path)


def main_parallel():
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        raise Exception("Expected \'--\' followed by arguments to the script")

    parser = argparse.ArgumentParser("prepare_coseg")
    parser.add_argument('--shapes-dir', type=str, required=True, help='Path to raw shapes directory with .obj files to normalize, the files will be overwritten!')
    args = parser.parse_args(argv)
    shapes_dir = Path(args.shapes_dir)
    print(f"Normalizing obj files in [{shapes_dir}]")
    cpu_count = multiprocessing.cpu_count()  # 20
    p = multiprocessing.Pool(cpu_count)
    obj_file_paths = list(shapes_dir.glob("*.obj"))
    p.map(normalize_obj, obj_file_paths)
    p.close()
    p.join()


if __name__ == "__main__":
    """
    Please note that this will overwrite the files
    """
    main_parallel()
