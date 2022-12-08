#!/usr/bin/env python3

import json
import random
import argparse
import traceback
import multiprocessing
import subprocess
from subprocess import Popen
from functools import partial
from pathlib import Path


def calculate_stability_proc(pred_obj_file_path, apply_normalization, skip_components_check, blender_exe: Path):
    print(f"Calculating stability for object [{pred_obj_file_path}]")
    simulation_blend_file_path = Path(__file__).parent.joinpath('stability_simulation.blend').resolve()
    stability_script_path = Path(__file__).parent.joinpath('stability.py').resolve()
    cmd = [str(blender_exe.expanduser()),
           str(simulation_blend_file_path),
           '-b', '--python', str(stability_script_path), '--',
           'sim-obj', '--obj-path', str(pred_obj_file_path)]
    if apply_normalization:
        cmd.append('--apply-normalization')
    if skip_components_check:
        cmd.append('--skip-components-check')
    print(" ".join(cmd))
    process = Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = process.communicate()
    result = out.splitlines()
    score_str_list = [line for line in result if 'Score [' in line]
    structurally_valid_str_list = [line for line in result if 'is_structurally_valid' in line]
    assert score_str_list, out
    score = float(score_str_list[0][-8:-1])
    assert structurally_valid_str_list, out
    is_structurally_valid = True if 'True' in structurally_valid_str_list[0] else False
    assert score > 0.1
    return score, is_structurally_valid


def sim_dir_parallel(args):
    cpu_count = multiprocessing.cpu_count()
    stability_json = {}
    count_stable = 0
    count_structurally_valid = 0
    count_good = 0
    dir_path = Path(args.dir_path).resolve()
    print(f"Calculating stability for dir [{dir_path}] with [{cpu_count}] processes")
    try:
        obj_files = sorted(dir_path.glob("*.obj"))
        print(len(obj_files))
        if args.limit and args.limit < len(obj_files):
            obj_files = random.sample(obj_files, args.limit)
        blender_exe = Path(args.blender_exe).resolve()
        calculate_stability_proc_partial = partial(calculate_stability_proc,
                                                   apply_normalization=args.apply_normalization,
                                                   skip_components_check=args.skip_components_check,
                                                   blender_exe=blender_exe)
        p = multiprocessing.Pool(cpu_count)
        stability_results = p.map(calculate_stability_proc_partial, obj_files)
        p.close()
        p.join()
        for obj_file_idx, obj_file in enumerate(obj_files):
            stability_json[str(obj_file)] = stability_results[obj_file_idx]
            score = stability_results[obj_file_idx][0]
            is_structurally_valid = stability_results[obj_file_idx][1]
            count_stable += 1 if score > 0.98 else 0
            count_structurally_valid += 1 if is_structurally_valid else 0
            count_good += 1 if (score > 0.98 and is_structurally_valid) else 0
    except Exception as e:
        print(traceback.format_exc())
        print(repr(e))
    sample_count = len(stability_json)
    print(f"# stable samples [{count_stable}] out of total [{sample_count}]")
    print(f"# structurally valid samples [{count_structurally_valid}] out of total [{sample_count}]")
    print(f"# good samples [{count_good}] out of total [{sample_count}] = [{(count_good/sample_count) * 100}%]")
    # save the detailed results to a json file
    stability_json['execution-details'] = {}
    stability_json['execution-details']['dir-path'] = str(dir_path)
    json_result_file_path = Path(__file__).parent.joinpath('stability_results.json').resolve()
    with open(json_result_file_path, 'w') as json_result_file:
        json.dump(stability_json, json_result_file)
    print(f"Results per .obj file were saved to [{json_result_file_path}]")


def main():
    parser = argparse.ArgumentParser(prog='stability_parallel')
    parser.add_argument('--dir-path', type=str, required=True, help='Path to the dir to test with the \'stability metric\'')
    parser.add_argument('--blender-exe', type=str, required=True, help='Path to Blender executable')
    parser.add_argument('--skip-components-check', action='store_true', default=False, help='Skip checking if the shape is structurally valid')
    parser.add_argument('--apply-normalization', action='store_true', default=False, help='Apply normalization on the imported objects')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of shapes that will be evaluated, randomly selected shapes will be tested')

    try:
        args = parser.parse_args()
        sim_dir_parallel(args)
    except Exception as e:
        print(repr(e))
        print(traceback.format_exc())
    

if __name__ == '__main__':
    main()
