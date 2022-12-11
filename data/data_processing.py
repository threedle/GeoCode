import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import traceback
import torch
from common.file_util import load_obj
from common.point_cloud_util import normalize_point_cloud


def generate_point_clouds(data_dir, phase, num_points, num_point_clouds_per_combination,
                          processed_dataset_dir_name, sampling_method,
                          gaussian=0.0, apply_point_cloud_normalization=False):
    """
    samples point cloud from mesh
    """
    dataset_dir = Path(data_dir, phase)
    processed_dataset_dir = dataset_dir.joinpath(processed_dataset_dir_name)
    processed_dataset_dir.mkdir(exist_ok=True)
    files = sorted(dataset_dir.glob('*.obj'))
    for file in tqdm(files):
        faces = None
        vertices = None
        should_load_obj = True  # this is done to only load the obj if it is actually required
        for point_cloud_idx in range(num_point_clouds_per_combination):
            new_file_name = Path(processed_dataset_dir, file.with_suffix('.npy').name.replace(".npy", f"_{point_cloud_idx}.npy"))
            if new_file_name.is_file():
                continue
            # only load the obj (once per all the instances) if it is actually needed
            if should_load_obj:
                vertices, faces = load_obj(file)
                vertices = vertices.reshape(1, vertices.shape[0], vertices.shape[1])
                vertices = torch.from_numpy(vertices)
                faces = torch.from_numpy(faces)
                should_load_obj = False

            try:
                point_cloud = sampling_method(faces, vertices, num_points=num_points)
            except Exception as e:
                print(traceback.format_exc())
                print(repr(e))
                print(file)
                continue
            if apply_point_cloud_normalization:
                # normalize the point cloud and use center of bounding box
                point_cloud = normalize_point_cloud(point_cloud)
            if gaussian and gaussian > 0.0:
                point_cloud += np.random.normal(0, gaussian, point_cloud.shape)
            np.save(str(new_file_name), point_cloud)


def normalize_labels(data_dir, phase, processed_dataset_dir_name, params_descriptors, train_with_visibility_label):
    dataset_dir = Path(data_dir, phase)
    processed_dataset_dir = dataset_dir.joinpath(processed_dataset_dir_name)
    processed_dataset_dir.mkdir(exist_ok=True)

    files = sorted(dataset_dir.glob('*.yml'))
    for file in files:
        if not file.is_file():
            # it is only allowed to not have a gt yml file when we are in test phase
            assert phase == "test"
            continue
        save_path = Path(processed_dataset_dir, file.name)
        if save_path.is_file():
            # this will skip normalization if the file exists, but if the recipe file changes, then normalization needs to be performed again
            # in that case, disable this if statement to regenerate the normalized labels
            continue
        with open(file, 'r') as f:
            yml_obj = yaml.load(f, Loader=yaml.FullLoader)
        normalized_yml_obj = yml_obj.copy()

        # only apply the normalization to the inputs that were changed in this dataset
        for param_name, param_descriptor in params_descriptors.items():
            param_input_type = param_descriptor.input_types
            min_val = param_descriptor.min_val
            max_val = param_descriptor.max_val
            if param_input_type == 'Integer':
                normalized_yml_obj[param_name] -= min_val
            elif param_input_type == 'Float':
                value = yml_obj[param_name]
                normalized_yml_obj[param_name] = (value - min_val) / (max_val - min_val)
            elif param_input_type == 'Boolean':
                pass
            elif param_input_type == 'Vector':
                param_name_no_axis = param_name[:-2]
                for axis in ['x', 'y', 'z']:
                    if param_name[-2:] != f" {axis}":
                        continue
                    value = yml_obj[param_name_no_axis][axis]
                    normalized_yml_obj[param_name_no_axis][axis] = (value - min_val) / (max_val - min_val)
            if train_with_visibility_label and not params_descriptors[param_name].is_visible(yml_obj):
                normalized_yml_obj[param_name] = -1

        with open(save_path, 'w') as out_file:
            yaml.dump(normalized_yml_obj, out_file)
