import bpy
import numpy as np
from common.file_util import load_obj
from common.bpy_util import select_shape
from common.intersection_util import isolate_node_as_final_geometry


def triangle_area(x):
    a = x[:, 0, :] - x[:, 1, :]
    b = x[:, 0, :] - x[:, 2, :]
    cross = np.cross(a, b)
    area = 0.5 * np.norm(cross, dim=1)
    return area


def object_sanity_check(obj_file):
    try:
        vertices, faces = load_obj(obj_file)
        vertices = vertices.reshape(1, vertices.shape[0], vertices.shape[1])
        faces = vertices.squeeze()[faces]
        triangle_area(faces)
    except Exception:
        print('Invalid sample')
        return False
    return True


def validate_monoleg(node_label, factor=0.08):
    chair = select_shape()
    revert_isolation = isolate_node_as_final_geometry(chair, node_label)

    dup_obj = chair.copy()
    dup_obj.data = chair.data.copy()
    dup_obj.animation_data_clear()
    bpy.context.collection.objects.link(dup_obj)
    # move for clarity
    dup_obj.location.x += 2.0
    # set active
    bpy.ops.object.select_all(action='DESELECT')
    dup_obj.select_set(True)
    bpy.context.view_layer.objects.active = dup_obj
    # apply the modifier to turn the geometry node to a mesh
    bpy.ops.object.modifier_apply(modifier="GeometryNodes")
    # export the object
    assert dup_obj.type == 'MESH'

    revert_isolation()

    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    center_of_volume = dup_obj.location[2]
    # another option is to use (type='ORIGIN_CENTER_OF_MASS', center='MEDIAN') as the center of mass
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    center_of_mass = dup_obj.location[2]
    height = dup_obj.dimensions[2]

    if center_of_volume - center_of_mass > factor * height:
        return True
    return False
