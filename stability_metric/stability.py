#!/usr/bin/env python3

import bpy
import sys
import argparse
import numpy as np
import traceback
from mathutils import Vector

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

from common.bpy_util import normalize_scale, select_obj
from common.intersection_util import detect_cross_intersection


class DropSimulator:
    def __init__(self, args):
        self.drop_height = 0.1
        self.duration_sec = 5
        self.skip_components_check = args.skip_components_check
        self.apply_normalization = args.apply_normalization
        self.obj_file_path = Path(args.obj_path).expanduser()

    def simulate(self):
        print(f"Importing object file [{self.obj_file_path}]")
        bpy.ops.import_scene.obj(filepath=str(self.obj_file_path), use_split_objects=False)
        obj = bpy.context.selected_objects[0]
        obj.data.materials.clear()
        select_obj(obj)
        if self.apply_normalization:
            print("Normalizing object...")
            normalize_scale(obj)
        # set origin to center of mass
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
        vertices = np.array([(obj.matrix_world @ v.co) for v in obj.data.vertices])
        # verify that the object is normalized
        max_diff = 0.05
        max_dist_from_center = abs(1.0 - np.max(np.sqrt(np.sum((vertices ** 2), axis=1))))
        assert max_dist_from_center < max_diff, f"Point cloud is not normalized [{max_dist_from_center} > {max_diff}] for sample [{self.obj_file_path.name}]. If this is an external dataset, please consider adding --apply-normalization flag."
        # position the object at drop height
        obj.location = Vector((0, 0, obj.location.z - min(vertices[:, 2]) + self.drop_height))
        height_before_drop = max(vertices[:, 2]) - min(vertices[:, 2])

        # apply rigid body simulation
        bpy.ops.rigidbody.object_add()
        frame_end = self.duration_sec * 25
        area = [a for a in bpy.context.screen.areas if a.type == "VIEW_3D"][0]
        with bpy.context.temp_override(area=area):
            select_obj(obj)
            bpy.ops.rigidbody.bake_to_keyframes(frame_start=1, frame_end=frame_end, step=1)
            bpy.context.scene.frame_current = frame_end
            obj.data.update()
            bpy.context.view_layer.update()
            print("Simulation completed")
            self.eval(obj, height_before_drop)

    def eval(self, obj, height_before_drop):
        vertices = np.array([(obj.matrix_world @ v.co) for v in obj.data.vertices])
        height_after_drop = max(vertices[:, 2]) - min(vertices[:, 2])
        score = min(height_after_drop, height_before_drop) / max(height_after_drop, height_before_drop)
        score = 1.0 if score > 1.0 else score
        print(f"Height before simulation [{height_before_drop:.5f}]")
        print(f"Height after simulation [{height_after_drop:.5f}]")
        print(f"Score [{score:.5f}]")
        print(f"is_stable (score > 0.98) [{score > 0.98}]")
        self.reset_simulation(obj)
        # structural evaluation
        if self.skip_components_check:
            print("is_structurally_valid (shape is connected) [True] (check skipped)")
        else:
            print("Checking structural validity...")
            obj_is_valid = self.is_structurally_connected()
            print(f"is_structurally_valid (shape is connected) [{obj_is_valid}]")

    def is_structurally_connected(self):
        """
        return True if all the parts that make the shape are reachable from any other part
        two parts are connected if they are intersecting or there is a path from one part
        to the other that passes only through intersecting parts
        """
        bpy.ops.import_scene.obj(filepath=str(self.obj_file_path), use_split_objects=False)
        obj = bpy.context.selected_objects[0]
        select_obj(obj)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')
        parts = bpy.context.selected_objects
        # in the beginning, each part is put into a separate set
        components = []
        for part in parts:
            components.append({part})

        idx_a = 0
        while idx_a + 1 < len(components):
            component_a = components[idx_a]
            found_intersection = False
            for idx_b in range(idx_a + 1, len(components)):
                component_b = components[idx_b]
                for part_a in component_a:
                    for part_b in component_b:
                        assert part_a.name != part_b.name
                        if len(detect_cross_intersection(part_a, part_b)) > 0:
                            components.remove(component_a)
                            components.remove(component_b)
                            components.append(component_a.union(component_b))
                            found_intersection = True
                            break
                    if found_intersection:
                        break
                if found_intersection:
                    break
            if not found_intersection:
                idx_a += 1
                # note that we can 'break' here and return False if we are only looking to have a single connected component
        bpy.ops.object.delete()
        return len(components) <= 1

    @staticmethod
    def reset_simulation(obj):
        bpy.context.scene.frame_current = 0
        obj.data.update()
        bpy.context.view_layer.update()
        bpy.ops.object.delete()


def main():
    if '--' in sys.argv:
        # refer to https://b3d.interplanety.org/en/how-to-pass-command-line-arguments-to-a-blender-python-script-or-add-on/
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        raise Exception("Expected \'--\' followed by arguments to the script")

    parser = argparse.ArgumentParser(prog='stability')
    parser.add_argument('--obj-path', type=str, required=True, help='Path to the object to test')
    parser.add_argument('--apply-normalization', action='store_true', default=False, help='Apply normalization on the object upon importing')
    parser.add_argument('--skip-components-check', action='store_true', default=False, help='Do not check that the shape is structurally valid')

    try:
        args = parser.parse_known_args(argv)[0]
        drop_simulator = DropSimulator(args)
        drop_simulator.simulate()
    except Exception as e:
        print(repr(e))
        print(traceback.format_exc())


if __name__ == '__main__':
    main()
