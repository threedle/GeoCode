#!/usr/bin/env python3

import shutil
import argparse
import multiprocessing
from pathlib import Path
from functools import partial
from subprocess import Popen, PIPE


def simplify_and_save_obj_process(yml_file_path: Path, dst_phase_dir: Path, simplification_ratio: float, recipe_file_path,
                                  blender_exe: Path, blend_file: Path):
    assert 0.0 <= simplification_ratio <= 1.0
    simplification_ratio_str = f'{simplification_ratio:.3f}'.replace('.', '_')
    out_file_name_no_ext = f'{yml_file_path.stem}_simplification_ratio_{simplification_ratio_str}'
    print(f"Converting [{yml_file_path}] to obj file [{out_file_name_no_ext}]")
    new_yml_path = dst_phase_dir.joinpath('yml_gt', f'{out_file_name_no_ext}.yml')
    new_obj_path = dst_phase_dir.joinpath('obj_gt', f'{out_file_name_no_ext}.obj')
    shutil.copy(yml_file_path, new_yml_path)
    save_obj_script_path = Path(__file__).parent.joinpath('..', 'common', 'save_obj.py').resolve()
    cmd = [str(blender_exe.expanduser()), str(blend_file.expanduser()), '-b', '--python',
           str(save_obj_script_path), '--',
           '--recipe-file-path', str(recipe_file_path),
           '--yml-file-path', str(yml_file_path),
           '--target-obj-file-path', str(new_obj_path),
           '--simplification-ratio', str(simplification_ratio)]
    print(" ".join(cmd))
    process = Popen(cmd, stdout=PIPE)
    process.wait()


def main():
    parser = argparse.ArgumentParser("simplified_mesh_dataset")
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--src-phase', type=str, required=True, help='Directory name of the source dataset phase')
    parser.add_argument('--dst-phase', type=str, default='simplified', help='Directory name of the destination dataset phase')
    parser.add_argument('--blender-exe', type=str, required=True, help='Path to blender executable')
    parser.add_argument('--blend-file', type=str, required=True, help='Path to blend file')
    parser.add_argument('--simplification-ratios', type=float, required=True, nargs='+', help='List of simplification ratios to generate datasets for,'
                                                                                              'note that 1.0 means the original shape is used with no simplification')
    args = parser.parse_args()

    assert all([0.0 <= simplification_ratio <= 1.0 for simplification_ratio in args.simplification_ratios])

    ds_dir = Path(args.dataset_dir).expanduser()
    src_phase_dir = ds_dir.joinpath(args.src_phase)
    dst_phase_dir = ds_dir.joinpath(args.dst_phase)
    if dst_phase_dir.is_dir():
        raise Exception(f'Simplified mesh dataset target directory [{dst_phase_dir}] already exists')
    dst_phase_dir.mkdir()

    dest_yml_gt_path = dst_phase_dir.joinpath('yml_gt')
    dest_yml_gt_path.mkdir(exist_ok=True)
    dest_obj_gt_path = dst_phase_dir.joinpath('obj_gt')
    dest_obj_gt_path.mkdir(exist_ok=True)

    blender_exe = Path(args.blender_exe)
    blend_file = Path(args.blend_file)

    # create all the obj from the prediction yaml files
    # note that for point cloud we have one yml and for sketch we have multiple yml files (one for each camera angle)
    cpu_count = multiprocessing.cpu_count()
    print(f"Generating simplified-mesh dataset in [{src_phase_dir}] with [{cpu_count}] processes")
    recipe_file_path = Path(args.dataset_dir, 'recipe.yml')
    src_yml_gt_path = src_phase_dir.joinpath('yml_gt')
    yml_file_paths = [yml_file_path for yml_file_path in src_yml_gt_path.glob("*.yml")]
    # note that 1.0 means no simplification takes place
    for simplification_ratio in args.simplification_ratios:
        print(f"Started generating simplified-mesh dataset with ratio [{simplification_ratio}]")
        simplify_and_save_obj_process_partial = partial(simplify_and_save_obj_process,
                                                        dst_phase_dir=dst_phase_dir,
                                                        simplification_ratio=simplification_ratio,
                                                        recipe_file_path=recipe_file_path,
                                                        blender_exe=blender_exe,
                                                        blend_file=blend_file)
        p = multiprocessing.Pool(cpu_count)
        p.map(simplify_and_save_obj_process_partial, yml_file_paths)
        p.close()
        p.join()


if __name__ == "__main__":
    main()
