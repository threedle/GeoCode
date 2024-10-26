#!/usr/bin/env python3

import sys
import bpy
import copy
import time
import yaml
import json
import hashlib
import argparse
import traceback
import subprocess
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

from common.param_descriptors import ParamDescriptors
from common.file_util import get_recipe_yml_obj, hash_file_name, get_source_recipe_file_path, save_yml
from common.bpy_util import refresh_obj_in_viewport, select_objs, select_shape, get_geometric_nodes_modifier, save_obj
from common.domain import Domain
from common.input_param_map import get_input_param_map, randomize_all_params, yml_to_shape
from dataset_generator.shape_validators.common_validations import object_sanity_check
from dataset_generator.shape_validators.shape_validator_factory import ShapeValidatorFactory


def shape_to_yml(gnodes_mod):
    shape_yml_obj = {}
    # get an arbitrary "Group Input" node
    group_input_nodes = [node for node in gnodes_mod.node_group.nodes if node.type == 'GROUP_INPUT']
    assert len(group_input_nodes) > 0
    group_input_node = group_input_nodes[0]
    for input in group_input_node.outputs:
        param_name = str(input.name)
        if len(param_name) == 0:
            continue
        param_val = gnodes_mod[input.identifier]
        if input.bl_label == "Vector":
            shape_yml_obj[param_name] = {}
            shape_yml_obj[param_name]['x'] = param_val[0]
            shape_yml_obj[param_name]['y'] = param_val[1]
            shape_yml_obj[param_name]['z'] = param_val[2]
        else:
            shape_yml_obj[param_name] = param_val
    return shape_yml_obj


def save_obj_label(gnodes_mod, target_yml_file_path: Path):
    """
    convert the object to parameter space and save it as a yaml file
    """
    shape_yml_obj = shape_to_yml(gnodes_mod)
    save_yml(shape_yml_obj, target_yml_file_path)


def update_base_shape_in_yml(gnodes_mod, recipe_file_path: Path):
    """
    This will completely overwrite the base shape in the given yml file while keeping other
    fields in the yaml untouched.
    If the yml file does not exist, it will generate a new base shape as yaml in the given file.
    """
    print(f'Updating the base shape in the YML file [{recipe_file_path}]')
    # init an empty object in case the file does not exist
    recipe_yml_obj = {}
    if recipe_file_path.is_file():
        recipe_yml_obj = get_recipe_yml_obj(recipe_file_path)
    base_yml_obj = shape_to_yml(gnodes_mod)
    # completely overwrite 'base' in the final yml object
    recipe_yml_obj['base'] = base_yml_obj
    # save the object as YML file
    save_yml(recipe_yml_obj, recipe_file_path)
    return recipe_yml_obj


def json_hash(json_obj):
    return hashlib.md5(json.dumps(json_obj).encode("utf-8")).hexdigest().strip()


def update_recipe_yml_obj_with_metadata(recipe_yml_obj, gnodes_mod, write_dataset_generation=False):
    # loops through all the inputs in the geometric node group
    data_types = {}
    group_input_nodes = [node for node in gnodes_mod.node_group.nodes if node.type == 'GROUP_INPUT']
    assert len(group_input_nodes) > 0
    group_input_node = group_input_nodes[0]
    for input in group_input_node.outputs:
        param_name = str(input.name)
        if len(param_name) == 0:
            continue
        data_types[param_name] = {}
        data_types[param_name]['type'] = input.bl_label
        if input.bl_label != 'Boolean':
            data_types[param_name]['min'] = gnodes_mod.node_group.interface.items_tree[param_name].min_value
            data_types[param_name]['max'] = gnodes_mod.node_group.interface.items_tree[param_name].max_value
    recipe_yml_obj['data_types'] = data_types
    if write_dataset_generation:
        dataset_generation = copy.deepcopy(data_types)
        for param_name in dataset_generation:
            if dataset_generation[param_name]['type'] not in ['Boolean', 'Integer']:
                dataset_generation[param_name]['samples'] = 5
            del dataset_generation[param_name]['type']
        recipe_yml_obj['dataset_generation'] = dataset_generation


def generate_dataset(domain, dataset_dir: Path, phase, random_shapes_per_value, parallel=1, mod=None):
    """
    Params:
      random_shapes_per_value - number of random shapes we will generate per parameter value
      mod - allows running this in parallel
    """
    try:
        # all other processes must wait for the folder to be created before continuing
        phase_dir = dataset_dir.joinpath(phase)
        yml_gt_dir = phase_dir.joinpath('yml_gt')
        obj_gt_dir = phase_dir.joinpath('obj_gt')
        if parallel > 1 and mod != 0:
            while not (dataset_dir.is_dir() and phase_dir.is_dir() and yml_gt_dir.is_dir() and obj_gt_dir.is_dir()):
                time.sleep(2)

        dataset_dir.mkdir(exist_ok=True)
        phase_dir.mkdir(exist_ok=True)

        obj = select_shape()
        # get the geometric nodes modifier for the object
        gnodes_mod = get_geometric_nodes_modifier(obj)
        recipe_file_path = get_source_recipe_file_path(domain)
        if parallel <= 1:
            update_base_shape_in_yml(gnodes_mod, recipe_file_path)

        # load recipe file and add some required metadata to it
        recipe_yml_obj = get_recipe_yml_obj(recipe_file_path)
        base_shape_yml = recipe_yml_obj['base'].copy()  # used to return the viewport shape to this base shape at the end of the dataset generation
        update_recipe_yml_obj_with_metadata(recipe_yml_obj, gnodes_mod)

        if parallel <= 1:
            # save the recipe object as yml file in the dataset main dir (since it now also contains additional required metadata)
            target_recipe_file_path = dataset_dir.joinpath('recipe.yml')
            save_yml(recipe_yml_obj, target_recipe_file_path)

        yml_gt_dir.mkdir(exist_ok=True)
        obj_gt_dir.mkdir(exist_ok=True)

        input_params_map = get_input_param_map(gnodes_mod, recipe_yml_obj)
        inputs_to_eval = list(input_params_map.keys())
        param_descriptors = ParamDescriptors(recipe_yml_obj, inputs_to_eval)
        param_descriptors_map = param_descriptors.get_param_descriptors_map()
        dup_hashes_attempts = []

        shape_validator = ShapeValidatorFactory.create_validator(domain)

        num_disqualified = 0
        num_intersections = 0
        
        existing_samples = {}
        for curr_param_name, curr_input_param in input_params_map.items():
            for value_idx, curr_param_value in enumerate(curr_input_param.possible_values):
                shape_idx = 0
                while shape_idx < random_shapes_per_value:
                    if parallel > 1:
                        if hash_file_name(f'{curr_param_name}_{value_idx}_{shape_idx}') % parallel != mod:
                            shape_idx += 1
                            continue

                    curr_param_value_str_for_file = f"{curr_param_value:.4f}".replace('.', '_')
                    file_name = f"{domain}_{curr_input_param.get_name_for_file()}_{curr_param_value_str_for_file}_{shape_idx:04d}"
                    obj_file = obj_gt_dir.joinpath(f"{file_name}.obj")
                    yml_file = yml_gt_dir.joinpath(f"{file_name}.yml")
                    if obj_file.is_file() and yml_file.is_file() and object_sanity_check(obj_file):
                        with open(yml_file, 'r') as file:
                            param_values_map_from_yml = yaml.load(file, Loader=yaml.FullLoader)
                        param_values_map = {}
                        for param_name, _ in input_params_map.items():
                            if param_name[-2:] in [' x', ' y', ' z']:
                                param_values_map[param_name] = param_values_map_from_yml[param_name[:-2]][param_name[-1:]]
                            else:
                                param_values_map[param_name] = param_values_map_from_yml[param_name]
                            
                        shape_yml = {}
                        for param_name, param_value in param_values_map.items():
                            if not param_descriptors_map[param_name].is_visible(param_values_map):
                                shape_yml[param_name] = -1
                            else:
                                shape_yml[param_name] = round(param_value, 4)
                        sample_hash = json_hash(shape_yml)
                        if sample_hash in existing_samples:
                            raise Exception("Found a duplicate within a single process")
                        existing_samples[sample_hash] = file_name
                        shape_idx += 1
                        continue

                    param_values_map = randomize_all_params(input_params_map)
                    param_values_map[curr_param_name] = curr_param_value

                    if not param_descriptors.check_constraints(param_values_map):
                        num_disqualified += 1
                        with open(f'{dataset_dir}/retry_{mod}.log', 'a') as f:
                            f.write(f'constraints {file_name}\n')
                        continue

                    if not param_descriptors_map[curr_param_name].is_visible(param_values_map):
                        with open(f'{dataset_dir}/retry_{mod}.log', 'a') as f:
                            f.write(f'visibility conditions {file_name} [{param_descriptors_map[curr_param_name].visibility_condition}] \n')
                        continue

                    for param_name, param_value in param_values_map.items():
                        input_params_map[param_name].assign_value(param_value)

                    # sanity check to make sure we did not override what we are trying to generate
                    assert abs(input_params_map[curr_param_name].get_value() - curr_param_value) < 1e-6
                    # must refresh the shape at this point
                    refresh_obj_in_viewport(obj)

                    # shape-specific validation
                    is_valid, msg = shape_validator.validate_shape(input_params_map)
                    if not is_valid:
                        num_disqualified += 1
                        if 'intersect' in msg.lower():
                            num_intersections += 1
                        with open(f'{dataset_dir}/retry_{mod}.log', 'a') as f:
                            f.write(f'Shape invalid with message [{msg}] for file {file_name}\n')
                        continue

                    # make sure this is a completely new shape
                    shape_yml = {}
                    for param_name, param_value in param_values_map.items():
                        if not param_descriptors_map[param_name].is_visible(param_values_map):
                            shape_yml[param_name] = -1
                        else:
                            shape_yml[param_name] = round(input_params_map[param_name].get_value(), 4)
                    sample_hash = json_hash(shape_yml)
                    if sample_hash in existing_samples:
                        dup_hashes_attempts.append(sample_hash)
                        with open(f'{dataset_dir}/retry_{mod}.log', 'a') as f:
                            f.write(f'already exists {sample_hash} {file_name}\n')
                        continue
                    existing_samples[sample_hash] = file_name

                    target_yml_file_path = yml_gt_dir.joinpath(f"{file_name}.yml")
                    save_obj_label(gnodes_mod, target_yml_file_path)
                    target_obj_file_path = obj_gt_dir.joinpath(f"{file_name}.obj")
                    dup_obj = save_obj(target_obj_file_path)
                    # delete the duplicate object
                    select_objs(dup_obj)
                    bpy.ops.object.delete()
                    shape_idx += 1

        # return the shape in Blender viewport to its original state
        yml_to_shape(base_shape_yml, input_params_map, ignore_sanity_check=True)

        # log duplicate attempts
        if dup_hashes_attempts:
            dup_hashes_attempts_file_path = dataset_dir.joinpath(f"dup_hashes_attempts_{mod}.txt")
            with open(dup_hashes_attempts_file_path, 'a') as dup_hashes_attempts_file:
                dup_hashes_attempts_file.writelines([f"{h}\n" for h in dup_hashes_attempts])
                dup_hashes_attempts_file.write('---\n')

        existing_samples['metadata'] = {}
        existing_samples['metadata']['num_disqualified'] = num_disqualified
        existing_samples['metadata']['num_intersections'] = num_intersections
        return existing_samples

    except Exception as e:
        with open(f'{dataset_dir}/err_{mod}.log', 'a') as f:
            f.write(repr(e))
            f.write('\n')
            f.write(traceback.format_exc())
            f.write('\n\n')


def main_generate_dataset_single_proc(args, blender_exe, blend_file):
    assert blender_exe
    assert blend_file
    # show the main collections (if it is already shown, there is no effect)
    bpy.context.scene.view_layers['View Layer'].layer_collection.children['Main'].hide_viewport = False
    bpy.context.scene.view_layers['View Layer'].layer_collection.children['Main'].exclude = False

    try:
        dataset_dir = Path(args.dataset_dir).expanduser()
        existing_samples = generate_dataset(args.domain, dataset_dir, args.phase,
                                            args.num_variations, parallel=args.parallel, mod=args.mod)
        samples_hashes_file_path = dataset_dir.joinpath(f"sample_hashes_{args.mod}.json")
        with open(samples_hashes_file_path, 'w') as samples_hashes_file:
            json.dump(existing_samples, samples_hashes_file)
        print(f"Process [{args.mod}] done")
    except Exception as e:
        print(repr(e))
        print(traceback.format_exc())


def main_generate_dataset_parallel(args, blender_exe, blend_file):
    dataset_dir = Path(args.dataset_dir).expanduser()
    dataset_dir.mkdir(exist_ok=True)

    phase_dir = dataset_dir.joinpath(args.phase)
    phase_dir.mkdir(exist_ok=True)

    try:
        for existing_shapes_json_file_path in dataset_dir.glob("sample_hashes_*.json"):
            existing_shapes_json_file_path.unlink()

        # select the procedural shape in blender
        obj = select_shape()
        # get the geometric nodes modifier fo the object
        gnodes_mod = get_geometric_nodes_modifier(obj)
        recipe_file_path = get_source_recipe_file_path(args.domain)
        recipe_yml_obj = get_recipe_yml_obj(str(recipe_file_path))
        update_base_shape_in_yml(gnodes_mod, recipe_file_path)
        update_recipe_yml_obj_with_metadata(recipe_yml_obj, gnodes_mod)
        # save the recipe.yml file in the dataset's main dir (it now also contains required metadata)
        target_recipe_file_path = dataset_dir.joinpath('recipe.yml')
        save_yml(recipe_yml_obj, target_recipe_file_path)
        input_params_map = get_input_param_map(gnodes_mod, recipe_yml_obj)
        # loops through all the inputs in the geometric node group
        data_types = {}
        group_input_nodes = [node for node in gnodes_mod.node_group.nodes if node.type == 'GROUP_INPUT']
        assert len(group_input_nodes) > 0
        group_input_node = group_input_nodes[0]
        for input in group_input_node.outputs:
            param_name = str(input.name)
            if len(param_name) == 0:
                continue
            data_types[param_name] = {}
            data_types[param_name]['type'] = input.bl_label
            if input.bl_label != 'Boolean':
                data_types[param_name]['min'] = input.min_value
                data_types[param_name]['max'] = input.max_value
        recipe_yml_obj['data_types'] = data_types
        inputs_to_eval = list(input_params_map.keys())
        param_descriptors = ParamDescriptors(recipe_yml_obj, inputs_to_eval)

        expected_number_of_samples = param_descriptors.get_overall_num_of_classes_without_visibility_label() * args.num_variations
        print(f"Overall expected number of objects to generate is [{expected_number_of_samples}]")

        iteration_count = 0
        while True:
            iteration_count += 1
            duplicates = []

            # load any current saved state (this is to generate val, test, and train, since we cannot do this simultaneously)
            # please note that all the sample_hashes files are common to all phases of the dataset (val, test, and train) so we
            # will avoid creating the same sample across all of these phases

            existing_samples = {}
            num_disqualified = 0
            num_intersections = 0
            # add all the samples hashes from any other phase to avoid duplicates with other phases
            sample_hashes_json_file_path = dataset_dir.joinpath("sample_hashes.json")
            if sample_hashes_json_file_path.is_file():
                with open(sample_hashes_json_file_path, 'r') as existing_samples_file:
                    existing_samples = json.load(existing_samples_file)
                    if args.phase in existing_samples:
                        num_disqualified = existing_samples[args.phase]['metadata']['num_disqualified']
                        num_intersections = existing_samples[args.phase]['metadata']['num_intersections']
            # clear any current phase sample hashes as they are added by the processes
            existing_samples[args.phase] = {}

            # every process generates a 'sample_hashes_<id>.json' file containing hash -> file_name map
            for single_process_existing_samples_json_file_path in dataset_dir.glob("sample_hashes_*.json"):
                with open(single_process_existing_samples_json_file_path, 'r') as single_process_existing_samples_json_file:
                    single_process_existing_samples = json.load(single_process_existing_samples_json_file)
                    num_disqualified += single_process_existing_samples['metadata']['num_disqualified']
                    num_intersections += existing_samples[args.phase]['metadata']['num_intersections']
                    print(single_process_existing_samples_json_file_path)
                    print(single_process_existing_samples)
                    for hash, file_name in single_process_existing_samples.items():
                        if hash == 'metadata':
                            continue
                        is_dup = False
                        for phase, sample_hashes in existing_samples.items():
                            if hash in sample_hashes:
                                duplicates.append(file_name)
                                is_dup = True
                                break
                        if not is_dup:
                            existing_samples[args.phase][hash] = file_name

            existing_samples[args.phase]['metadata'] = {}
            existing_samples[args.phase]['metadata']['num_disqualified'] = num_disqualified
            existing_samples[args.phase]['metadata']['num_intersections'] = num_intersections

            # delete the duplicated files so they will be regenerated
            if duplicates:
                print(f"Found [{len(duplicates)}] duplicates that will be regenerated")
                print("\n\t".join(duplicates))
            for file_name in duplicates:
                obj_file = phase_dir.joinpath(file_name + ".obj")
                yml_file = phase_dir.joinpath(file_name + ".yml")
                obj_file.unlink()
                yml_file.unlink()

            # backup the current sample_hashes
            with open(f"{dataset_dir}/sample_hashes.json", 'w') as sample_hashes_file:
                json.dump(existing_samples, sample_hashes_file)

            # -1 is since we also store a metadata object for each phase
            if len(existing_samples[args.phase]) - 1 > expected_number_of_samples:
                raise Exception("Something went wrong, make sure you know how to count the number of expected samples")
            print(len(existing_samples[args.phase]))
            print(expected_number_of_samples)
            if len(existing_samples[args.phase]) - 1 == expected_number_of_samples:
                print('Done creating the requested dataset')
                break

            # since we need another iteration, we remove all the sample_hashes files
            for existing_samples_json_file_path in dataset_dir.glob("sample_hashes_*.json"):
                existing_samples_json_file_path.unlink()

            processes = []
            dataset_generator_path = Path(__file__).parent.joinpath('dataset_generator.py').resolve()
            for mod in range(args.parallel):
                try:
                    cmd = [str(blender_exe), str(blend_file), '-b', '--python', str(dataset_generator_path), '--',
                           'generate-dataset-single-process',
                           '--dataset-dir', str(dataset_dir),
                           '--domain', str(args.domain),
                           '--phase', args.phase,
                           '--num-variations', str(args.num_variations),
                           '--parallel', str(args.parallel),
                           '--mod', str(mod)]
                    print(f'Mod {mod}:')
                    print(" ".join(cmd))
                    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)  # DEVNULL is to avoid processes getting stuck
                    processes.append(process)
                except Exception as e:
                    print(repr(e))
                    print(traceback.format_exc())
            for process in processes:
                process.wait()
            
            print(f"Dataset generation iteration [{iteration_count}] is done")
    except Exception as e:
        print(repr(e))
        print(traceback.format_exc())


def main():
    if '--' in sys.argv:
        # refer to https://b3d.interplanety.org/en/how-to-pass-command-line-arguments-to-a-blender-python-script-or-add-on/
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        raise Exception("Expected \'--\' followed by arguments to the script")

    parser = argparse.ArgumentParser(prog='dataset_generator')
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--dataset-dir', type=str, required=True, help='Path to dataset directory')

    sp = parser.add_subparsers()
    sp_gen_single_proc = sp.add_parser('generate-dataset-single-process', parents=[common_parser])
    sp_gen_parallel = sp.add_parser('generate-dataset', parents=[common_parser])

    sp_gen_single_proc.set_defaults(func=main_generate_dataset_single_proc)
    sp_gen_single_proc.add_argument('--domain', type=Domain, choices=list(Domain), required=True, help='The domain name to generate the dataset for.')
    sp_gen_single_proc.add_argument('--phase', type=str, required=True, help='E.g. train, val, or test')
    sp_gen_single_proc.add_argument('--num-variations', type=int, default=3)
    sp_gen_single_proc.add_argument('--parallel', type=int, default=1, help='Number of processes that are running the script in parallel')
    sp_gen_single_proc.add_argument('--mod', type=int, default=None, help='The modulo for this process to match files\' hash')

    sp_gen_parallel.set_defaults(func=main_generate_dataset_parallel)
    sp_gen_parallel.add_argument('--domain', type=Domain, choices=list(Domain), required=True, help='The domain name to generate the dataset for.')
    sp_gen_parallel.add_argument('--phase', type=str, required=True, help='E.g. train, val, or test')
    sp_gen_parallel.add_argument('--num-variations', type=int, default=3, help='The number of random shapes to generate for each parameter value')
    sp_gen_parallel.add_argument('--parallel', type=int, default=1, help='Number of processes that will run the script')

    blender_exe_path = Path(sys.argv[0]).resolve()
    blend_file_path = Path(sys.argv[1]).resolve()
    try:
        args = parser.parse_known_args(argv)[0]
        args.func(args, blender_exe_path, blend_file_path)
    except Exception as e:
        print(repr(e))
        print(traceback.format_exc())

    
if __name__ == "__main__":
    main()
