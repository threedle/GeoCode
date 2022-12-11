import bpy
import array
import mathutils
from object_print3d_utils import mesh_helpers
from common.bpy_util import select_objs, select_shape, refresh_obj_in_viewport


def isolate_node_as_final_geometry(obj, node_label):
    gm = obj.modifiers.get("GeometryNodes")
    group_output_node = None
    node_to_isolate = None
    for n in gm.node_group.nodes:
        # print(n.name), print(n.type), print(dir(n))
        if n.type == 'GROUP_OUTPUT':
            group_output_node = n
        elif n.label == node_label:
            node_to_isolate = n
    if not node_to_isolate:
        raise Exception(f"Did not find any node with the label [{node_label}]")

    realize_instances_node = group_output_node.inputs[0].links[0].from_node
    third_to_last_node = realize_instances_node.inputs[0].links[0].from_node
    third_to_last_node_socket = None
    # to later revert this operation, we need to find the socket which is currently connected
    # this happens since the SWITCH node has multiple options, and each option translates to
    # a different output socket in the node (so there isn't just one socket as you would think)
    for i, socket in enumerate(third_to_last_node.outputs):
        if socket.is_linked:
            third_to_last_node_socket = i
            break
    node_group = next(m for m in obj.modifiers if m.type == 'NODES').node_group
    # find the output socket that actually is connected to something,
    # we do this since some nodes have multiple output sockets
    out_socket_idx = 0
    for out_socket_idx, out_socket in enumerate(node_to_isolate.outputs):
        if out_socket.is_linked:
            break
    node_group.links.new(node_to_isolate.outputs[out_socket_idx], realize_instances_node.inputs[0])
    def revert():
        node_group.links.new(third_to_last_node.outputs[third_to_last_node_socket], realize_instances_node.inputs[0])
        refresh_obj_in_viewport(obj)
    return revert


def detect_self_intersection(obj):
    """
    refer to:
    https://blenderartists.org/t/self-intersection-detection/671080
    documentation of the intersection detection method
    https://docs.blender.org/api/current/mathutils.bvhtree.html
    """
    if not obj.data.polygons:
        return array.array('i', ())

    bm = mesh_helpers.bmesh_copy_from_object(obj, transform=False, triangulate=False)
    tree = mathutils.bvhtree.BVHTree.FromBMesh(bm, epsilon=0.00001)

    overlap = tree.overlap(tree)
    faces_error = {i for i_pair in overlap for i in i_pair}
    return array.array('i', faces_error)


def find_self_intersections(node_label):
    # intersection detection
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
    assert dup_obj.type == 'MESH'

    intersections = detect_self_intersection(dup_obj)

    # delete the duplicate
    bpy.ops.object.delete()

    revert_isolation()

    # reselect the original object
    select_shape()

    return len(intersections)


def detect_cross_intersection(obj1, obj2):
    if not obj1.data.polygons or not obj2.data.polygons:
        return array.array('i', ())

    bm1 = mesh_helpers.bmesh_copy_from_object(obj1, transform=False, triangulate=False)
    tree1 = mathutils.bvhtree.BVHTree.FromBMesh(bm1, epsilon=0.00001)
    bm2 = mesh_helpers.bmesh_copy_from_object(obj2, transform=False, triangulate=False)
    tree2 = mathutils.bvhtree.BVHTree.FromBMesh(bm2, epsilon=0.00001)

    overlap = tree1.overlap(tree2)
    faces_error = {i for i_pair in overlap for i in i_pair}
    return array.array('i', faces_error)


def find_cross_intersections(node_label1, node_label2):
    # intersection detection
    chair = select_shape()
    revert_isolation = isolate_node_as_final_geometry(chair, node_label1)

    dup_obj1 = chair.copy()
    dup_obj1.data = chair.data.copy()
    dup_obj1.animation_data_clear()
    bpy.context.collection.objects.link(dup_obj1)
    # move for clarity
    dup_obj1.location.x += 2.0
    # set active
    bpy.ops.object.select_all(action='DESELECT')
    dup_obj1.select_set(True)
    bpy.context.view_layer.objects.active = dup_obj1
    # apply the modifier to turn the geometry node to a mesh
    bpy.ops.object.modifier_apply(modifier="GeometryNodes")
    # export the object
    assert dup_obj1.type == 'MESH'

    revert_isolation()

    chair = select_shape()
    revert_isolation = isolate_node_as_final_geometry(chair, node_label2)

    dup_obj2 = chair.copy()
    dup_obj2.data = chair.data.copy()
    dup_obj2.animation_data_clear()
    bpy.context.collection.objects.link(dup_obj2)
    # move for clarity
    dup_obj2.location.x += 2.0
    # set active
    bpy.ops.object.select_all(action='DESELECT')
    dup_obj2.select_set(True)
    bpy.context.view_layer.objects.active = dup_obj2
    # apply the modifier to turn the geometry node to a mesh
    bpy.ops.object.modifier_apply(modifier="GeometryNodes")
    # export the object
    assert dup_obj2.type == 'MESH'

    revert_isolation()

    intersections = detect_cross_intersection(dup_obj1, dup_obj2)

    # delete the duplicate
    select_objs(dup_obj1, dup_obj2)
    bpy.ops.object.delete()

    # reselect the original object
    select_shape()

    return len(intersections)
