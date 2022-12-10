import json
import torch
import shutil
import traceback
import numpy as np
import multiprocessing
from pathlib import Path
from functools import partial
import pytorch_lightning as pl
from subprocess import Popen, PIPE
from data.dataset_pc import DatasetPC
from data.dataset_sketch import DatasetSketch
from barplot_util import gen_and_save_barplot
from common.param_descriptors import ParamDescriptors
from geocode_util import InputType, get_inputs_to_eval, calc_prediction_vector_size
from geocode_model import Model
from torch.utils.data import DataLoader
from chamfer_distance import ChamferDistance as chamfer_dist
from common.sampling_util import sample_surface
from common.file_util import load_obj, get_recipe_yml_obj
from common.point_cloud_util import normalize_point_cloud


def sample_pc_random(obj_path, num_points=10_000, apply_point_cloud_normalization=False):
    """
    Chamfer evaluation
    """
    vertices, faces = load_obj(obj_path)
    vertices = vertices.reshape(1, vertices.shape[0], vertices.shape[1])
    vertices = torch.from_numpy(vertices)
    faces = torch.from_numpy(faces)
    point_cloud = sample_surface(faces, vertices, num_points=num_points)
    if apply_point_cloud_normalization:
        point_cloud = normalize_point_cloud(point_cloud)
    # assert that the point cloud is normalized
    max_dist_in_pc = torch.max(torch.sqrt(torch.sum((point_cloud ** 2), dim=1)))
    threshold = 0.1
    assert abs(1.0 - max_dist_in_pc) <= threshold, f"PC of obj [{obj_path}] is not normalized, max distance in PC was [{abs(1.0 - max_dist_in_pc)}] but required to be <= [{threshold}]."
    return point_cloud


def get_chamfer_distance(target_pc, gt_pc, device, num_points_in_pc, check_rot=False):
    """
    num_points_in_pc - for sanity check
    check_rot - was done for the tables since they are symmetric, and sketches are randomly flipped
                it is ok to leave it on for the vase and chair, just makes it slower
    """
    gt_pc = gt_pc.reshape(1, gt_pc.shape[0], gt_pc.shape[1])  # add batch dim
    target_pc = target_pc.reshape(1, target_pc.shape[0], target_pc.shape[1])  # add batch dim
    assert gt_pc.shape[1] == target_pc.shape[1] == num_points_in_pc
    dist1, dist2, idx1, idx2 = chamfer_dist(target_pc.float().to(device), gt_pc.float().to(device))
    chamfer_distance = (torch.sum(dist1) + torch.sum(dist2)) / (gt_pc.shape[1] * 2)
    if check_rot:
        rot_mat = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        target_pc_rot = torch.matmul(rot_mat, target_pc.squeeze().t()).t().unsqueeze(0)
        dist1, dist2, idx1, idx2 = chamfer_dist(target_pc_rot.float().to(device), gt_pc.float().to(device))
        chamfer_distance_rot = (torch.sum(dist1) + torch.sum(dist2)) / (gt_pc.shape[1] * 2)
        return torch.min(chamfer_distance, chamfer_distance_rot)
    return chamfer_distance


def save_as_obj_proc(pred_yml_file_path: Path, recipe_file_path: Path, results_dir: Path, out_dir: str, blender_exe: str, blend_file: str):
    target_obj_file_path = results_dir.joinpath(out_dir, f'{pred_yml_file_path.stem}.obj')
    print(f"Converting [{pred_yml_file_path}] to obj file [{target_obj_file_path}]")
    save_obj_script_path = Path(__file__).parent.joinpath('..', 'common', 'save_obj.py').resolve()
    cmd = [f'{str(Path(blender_exe).expanduser())}', f'{str(Path(blend_file).expanduser())}', '-b', '--python',
           f"{str(save_obj_script_path)}", '--',
           '--recipe-file-path', str(recipe_file_path),
           '--yml-file-path', str(pred_yml_file_path),
           '--target-obj-file-path', str(target_obj_file_path),
           '--ignore-sanity-check']
    print(" ".join(cmd))
    process = Popen(cmd, stdout=PIPE)
    process.wait()


def test(opt):
    recipe_file_path = Path(opt.dataset_dir, 'recipe.yml')
    print(recipe_file_path)
    if not recipe_file_path.is_file():
        raise Exception(f'No \'recipe.yml\' file found in path [{recipe_file_path}]')
    recipe_yml_obj = get_recipe_yml_obj(str(recipe_file_path))

    inputs_to_eval = get_inputs_to_eval(recipe_yml_obj)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    camera_angles_to_process = recipe_yml_obj['camera_angles_train'] + recipe_yml_obj['camera_angles_test']
    camera_angles_to_process = [f'{a}_{b}' for a, b in camera_angles_to_process]

    param_descriptors = ParamDescriptors(recipe_yml_obj, inputs_to_eval, opt.use_regression)
    param_descriptors_map = param_descriptors.get_param_descriptors_map()
    detailed_vec_size = calc_prediction_vector_size(param_descriptors_map)
    print(f"Prediction vector length is set to [{sum(detailed_vec_size)}]")

    # setup required dirs
    required_dirs = ['barplot',
                     'yml_gt', 'yml_predictions_pc', 'yml_predictions_sketch',
                     'obj_gt', 'obj_predictions_pc', 'obj_predictions_sketch',
                     'render_gt', 'render_predictions_pc', 'render_predictions_sketch',
                     'sketch_gt']
    test_dir = Path(opt.dataset_dir, opt.phase)
    test_dir_obj_gt = test_dir.joinpath('obj_gt')
    results_dir = test_dir.joinpath(f'results_{opt.exp_name}')
    results_dir.mkdir(exist_ok=True)
    for dir in required_dirs:
        results_dir.joinpath(dir).mkdir(exist_ok=True)

    # save the recipe to the results directory
    shutil.copy(recipe_file_path, results_dir.joinpath('recipe.yml'))

    # find the best checkpoint (the one with the highest epoch number out of the saved checkpoints)
    exp_dir = Path(opt.models_dir, opt.exp_name)
    best_model_and_highest_epoch = None
    highest_epoch = 0
    for ckpt_file in exp_dir.glob("*.ckpt"):
        file_name = ckpt_file.name
        if 'epoch' not in file_name:
            continue
        epoch_start_idx = file_name.find('epoch') + len('epoch')
        epoch = int(file_name[epoch_start_idx:epoch_start_idx + 3])
        if epoch > highest_epoch:
            best_model_and_highest_epoch = ckpt_file
            highest_epoch = epoch
    print(f'Best model with highest epoch is [{best_model_and_highest_epoch}]')

    batch_size = 1
    test_dataloaders = []
    test_dataloaders_types = []
    # pc
    if InputType.pc in opt.input_type:
        test_dataset_pc = DatasetPC(inputs_to_eval, device, param_descriptors_map,
                                    opt.dataset_dir, opt.phase, random_pc=opt.random_pc,
                                    gaussian=opt.gaussian, apply_point_cloud_normalization=opt.normalize_pc,
                                    scanobjectnn=opt.scanobjectnn, augment_with_random_points=opt.augment_with_random_points)
        test_dataloader_pc = DataLoader(test_dataset_pc, batch_size=batch_size, shuffle=False,
                                        num_workers=2, prefetch_factor=2)
        test_dataloaders.append(test_dataloader_pc)
        test_dataloaders_types.append('pc')
    # sketch
    if InputType.sketch in opt.input_type:
        test_dataset_sketch = DatasetSketch(inputs_to_eval, param_descriptors_map,
                                            camera_angles_to_process, opt.pretrained_vgg,
                                            opt.dataset_dir, opt.phase)
        test_dataloader_sketch = DataLoader(test_dataset_sketch, batch_size=batch_size, shuffle=False,
                                            num_workers=2, prefetch_factor=2)
        test_dataloaders.append(test_dataloader_sketch)
        test_dataloaders_types.append('sketch')

    pl_model = Model.load_from_checkpoint(str(best_model_and_highest_epoch), batch_size=1,
                                          param_descriptors=param_descriptors, results_dir=results_dir,
                                          test_dir=test_dir, models_dir=opt.models_dir,
                                          test_dataloaders_types=test_dataloaders_types, test_input_type=opt.input_type,
                                          exp_name=opt.exp_name)

    if True:
        trainer = pl.Trainer(gpus=1)
        trainer.test(model=pl_model, dataloaders=test_dataloaders, ckpt_path=best_model_and_highest_epoch)

        # save the validation and test bar-plots as image
        barplot_target_dir = results_dir.joinpath('barplot')
        for barplot_type in ['val', 'test']:
            barplot_json_path = Path(opt.models_dir, opt.exp_name, f'{barplot_type}_barplot_top_1.json')
            if not barplot_json_path.is_file():
                print(f"Could not find barplot [{barplot_json_path}] skipping copy")
                continue
            barplot_target_image_path = barplot_target_dir.joinpath(f'{barplot_type}_barplot.png')
            title = "Validation Accuracy" if barplot_type == 'val' else "Test Accuracy"
            gen_and_save_barplot(barplot_json_path, title, barplot_target_image_path=barplot_target_image_path)
            shutil.copy(barplot_json_path, barplot_target_dir.joinpath(barplot_json_path.name))

    gt_dir = results_dir.joinpath('yml_gt')
    model_predictions_pc_dir = results_dir.joinpath('yml_predictions_pc')
    model_predictions_sketch_dir = results_dir.joinpath('yml_predictions_sketch')
    file_names = sorted([f.stem for f in test_dir_obj_gt.glob("*.obj")])

    random_pc_dir = test_dir.joinpath("point_cloud_random")
    if opt.scanobjectnn:
        # [:-2] removed the _0 suffix
        file_names = [str(f.stem)[:-2] for f in random_pc_dir.glob("*.npy")]

    # create all the obj from the prediction yaml files
    # note that for pc we have one yml and for sketch we have multiple yml files (one for each camera angle)
    if True:
        cpu_count = multiprocessing.cpu_count()
        print(f"Converting yml files to obj files using [{cpu_count}] processes")
        for yml_dir, out_dir in [(gt_dir, 'obj_gt'), (model_predictions_pc_dir, 'obj_predictions_pc'), (model_predictions_sketch_dir, 'obj_predictions_sketch')]:
            try:
                # for each gt obj file we might have multiple yml files as predictions, like for the sketches
                yml_files = sorted(yml_dir.glob("*.yml"))
                # filter out existing
                yml_files_filtered = [yml_file for yml_file in yml_files if not results_dir.joinpath(out_dir, f'{yml_file.stem}.obj').is_file()]
                if out_dir == 'obj_gt' and not yml_files:
                    # COSEG (or any external ds for which we do not have ground truth yml files)
                    for obj_file in test_dir_obj_gt.glob("*.obj"):
                        print(f"shutil [{obj_file}]")
                        shutil.copy(obj_file, str(Path(results_dir, out_dir, f"{obj_file.stem}_gt.obj")))
                    continue
                save_as_obj_proc_partial = partial(save_as_obj_proc,
                                                   recipe_file_path=recipe_file_path,
                                                   results_dir=results_dir,
                                                   out_dir=out_dir,
                                                   blender_exe=opt.blender_exe,
                                                   blend_file=opt.blend_file)
                p = multiprocessing.Pool(cpu_count)
                p.map(save_as_obj_proc_partial, yml_files_filtered)
                p.close()
                p.join()
            except Exception as e:
                print(traceback.format_exc())
                print(repr(e))
        print("Done converting yml files to obj files")

    if True:
        num_points_in_pc_for_chamfer = 10000
        chamfer_json = {'pc': {}, 'sketch': {}}
        chamfer_summary_json = {'pc': {'chamfer_sum': 0.0, 'num_samples': 0}, 'sketch': {'chamfer_sum': 0.0, 'num_samples': 0}}

        for file_idx, file_name in enumerate(file_names): # for each unique test object
            # get ground truth point cloud (uniform)
            gt_file_name = file_name
            if "_decimate_ratio_0" in file_name:
                # edge case for comparing on the decimated ds
                gt_file_name = gt_file_name.replace("_decimate_ratio_0_100", "_decimate_ratio_1_000")
                gt_file_name = gt_file_name.replace("_decimate_ratio_0_010", "_decimate_ratio_1_000")
                gt_file_name = gt_file_name.replace("_decimate_ratio_0_005", "_decimate_ratio_1_000")
            assert "_decimate_ratio_0_100" not in gt_file_name
            assert "_decimate_ratio_0_010" not in gt_file_name
            assert "_decimate_ratio_0_005" not in gt_file_name

            if opt.scanobjectnn:
                random_pc_path = random_pc_dir.joinpath(f"{file_name}_0.npy")
                gt_pc = np.load(str(random_pc_path))
                gt_pc = torch.from_numpy(gt_pc).float()
                num_points_in_pc_for_chamfer = 2048
            else:
                gt_pc = sample_pc_random(results_dir.joinpath('obj_gt',f'{file_name}_gt.obj'),
                                         num_points=num_points_in_pc_for_chamfer,
                                         apply_point_cloud_normalization=opt.normalize_pc)

            for input_type, model_prediction_dir in [('pc', model_predictions_pc_dir), ('sketch', model_predictions_sketch_dir)]:
                yml_files = sorted(model_prediction_dir.glob(f"{file_name}_*.yml"))
                for yml_file in yml_files:
                    yml_file_base_name_no_ext = yml_file.stem
                    target_pc = sample_pc_random(results_dir.joinpath(f'obj_predictions_{input_type}', f'{yml_file_base_name_no_ext}.obj'),
                                                 num_points=num_points_in_pc_for_chamfer,
                                                 apply_point_cloud_normalization=opt.normalize_pc)
                    chamf_distance = get_chamfer_distance(target_pc, gt_pc, device, num_points_in_pc_for_chamfer, check_rot=(input_type == 'sketch'))
                    chamfer_summary_json[input_type]['chamfer_sum'] += chamf_distance.item()
                    chamfer_summary_json[input_type]['num_samples'] += 1
                    chamfer_json[input_type][yml_file_base_name_no_ext] = chamf_distance.item()

        # compute overall average
        if chamfer_summary_json['pc']['num_samples'] > 0:
            chamfer_json['pc']['avg'] = chamfer_summary_json['pc']['chamfer_sum'] / chamfer_summary_json['pc']['num_samples']
        if chamfer_summary_json['sketch']['num_samples'] > 0:
            chamfer_json['sketch']['avg'] = chamfer_summary_json['sketch']['chamfer_sum'] / chamfer_summary_json['sketch']['num_samples']

        # save chamfer json to the results dir
        with open(results_dir.joinpath("chamfer.json"), "w") as outfile:
            json.dump(chamfer_json, outfile)

        print("Done calculating Chamfer distances")
