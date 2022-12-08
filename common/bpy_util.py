import bpy
import math
from mathutils import Vector
from typing import Union
from pathlib import Path


def save_obj(target_obj_file_path: Union[Path, str], additional_objs_to_save=None, simplification_ratio=None):
    """
    save the object and returns a mesh duplicate version of it
    """
    obj = select_shape()
    refresh_obj_in_viewport(obj)
    dup_obj = copy(obj)
    # set active
    bpy.ops.object.select_all(action='DESELECT')
    dup_obj.select_set(True)
    bpy.context.view_layer.objects.active = dup_obj
    # apply the modifier to turn the geometry node to a mesh
    bpy.ops.object.modifier_apply(modifier="GeometryNodes")
    if simplification_ratio and simplification_ratio < 1.0:
        bpy.ops.object.modifier_add(type='DECIMATE')
        dup_obj.modifiers["Decimate"].decimate_type = 'COLLAPSE'
        dup_obj.modifiers["Decimate"].ratio = simplification_ratio
        bpy.ops.object.modifier_apply(modifier="Decimate")
    assert dup_obj.type == 'MESH'
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # set origin to center of bounding box
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    dup_obj.location.x = dup_obj.location.y = dup_obj.location.z = 0
    normalize_scale(dup_obj)
    if additional_objs_to_save:
        for additional_obj in additional_objs_to_save:
            additional_obj.select_set(True)
    # save
    bpy.ops.export_scene.obj(filepath=str(target_obj_file_path), use_selection=True, use_materials=False, use_triangles=True)
    return dup_obj


def get_geometric_nodes_modifier(obj):
    # loop through all modifiers of the given object
    gnodes_mod = None
    for modifier in obj.modifiers:
        # check if current modifier is the geometry nodes modifier
        if modifier.type == "NODES":
            gnodes_mod = modifier
            break
    return gnodes_mod


def normalize_scale(obj):
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # set origin to the center of the bounding box
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    obj.location.x = 0
    obj.location.y = 0
    obj.location.z = 0

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    max_vert_dist = math.sqrt(max([v.co.dot(v.co) for v in obj.data.vertices]))

    for v in obj.data.vertices:
        v.co /= max_vert_dist

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # verify that the shape is normalized
    # max_vert_dist = math.sqrt(max([v.co.dot(v.co) for v in obj.data.vertices]))
    # assert abs(max_vert_dist - 1.0) < 0.01


def setup_lights():
    """
    setup lights for rendering
    used for visualization of 3D objects as images
    """
    scene = bpy.context.scene
    # light 1
    light_data_1 = bpy.data.lights.new(name="light_data_1", type='POINT')
    light_data_1.energy = 300
    light_object_1 = bpy.data.objects.new(name="Light_1", object_data=light_data_1)
    light_object_1.location = Vector((10, -10, 10))
    scene.collection.objects.link(light_object_1)
    # light 2
    light_data_2 = bpy.data.lights.new(name="light_data_2", type='POINT')
    light_data_2.energy = 300
    light_object_2 = bpy.data.objects.new(name="Light_2", object_data=light_data_2)
    light_object_2.location = Vector((-10, -10, 10))
    scene.collection.objects.link(light_object_2)
    # light 3
    light_data_3 = bpy.data.lights.new(name="light_data_3", type='POINT')
    light_data_3.energy = 300
    light_object_3 = bpy.data.objects.new(name="Light_3", object_data=light_data_3)
    light_object_3.location = Vector((10, 0, 10))
    scene.collection.objects.link(light_object_3)


def look_at(obj_camera, point):
    """
    orient the given camera with a fixed position to loot at a given point in space
    """
    loc_camera = obj_camera.matrix_world.to_translation()
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj_camera.rotation_euler = rot_quat.to_euler()


def clean_scene(start_with_strings=["Camera", "procedural", "Light"]):
    """
    delete all object of which the name's prefix is matching any of the given strings
    """
    scene = bpy.context.scene
    bpy.ops.object.select_all(action='DESELECT')
    for obj in scene.objects:
        if any([obj.name.startswith(starts_with_string) for starts_with_string in start_with_strings]):
            # select the object
            if obj.visible_get():
                obj.select_set(True)
    bpy.ops.object.delete()


def del_obj(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.delete()


def refresh_obj_in_viewport(obj):
    # the following two line cause the object to update according to the new geometric nodes input
    obj.show_bounds = not obj.show_bounds
    obj.show_bounds = not obj.show_bounds


def select_objs(*objs):
    bpy.ops.object.select_all(action='DESELECT')
    for i, obj in enumerate(objs):
        if i == 0:
            bpy.context.view_layer.objects.active = obj
        obj.select_set(True)


def select_obj(obj):
    select_objs(obj)


def select_shape():
    """
    select the procedural shape in the blend file
    note that in all our domains, the procedural shape is named "procedural shape" within the blend file
    """
    obj = bpy.data.objects["procedural shape"]
    select_obj(obj)
    return obj


def copy(obj):
    dup_obj = obj.copy()
    dup_obj.data = obj.data.copy()
    dup_obj.animation_data_clear()
    bpy.context.collection.objects.link(dup_obj)
    return dup_obj
