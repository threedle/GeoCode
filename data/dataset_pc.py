import numpy as np
import torch.utils.data as data
import torch
import yaml
from pathlib import Path
from .data_processing import generate_point_clouds, normalize_labels
from common.sampling_util import sample_surface, farthest_point_sampling
from .dataset_util import assemble_targets


class DatasetPC(data.Dataset):
    def __init__(self,
                 inputs_to_eval,
                 dataset_processing_preferred_device,
                 params_descriptors,
                 data_dir,
                 phase,
                 num_points=1500,
                 num_point_clouds_per_combination=1,
                 random_pc=None,
                 gaussian=0.0,
                 apply_point_cloud_normalization=False,
                 scanobjectnn=False,
                 augment_with_random_points=True,
                 train_with_visibility_label=True):
        self.inputs_to_eval = inputs_to_eval
        self.data_dir = data_dir
        self.phase = phase
        self.random_pc = random_pc
        self.gaussian = gaussian
        self.apply_point_cloud_normalization = apply_point_cloud_normalization
        self.dataset_processing_preferred_device = dataset_processing_preferred_device
        self.train_with_visibility_label = train_with_visibility_label
        self.yml_gt_normalized_dir_name = 'yml_gt_normalized'
        self.point_cloud_fps_dir_name = 'point_cloud_fps'
        self.point_cloud_random_dir_name = 'point_cloud_random'
        self.num_point_clouds_per_combination = num_point_clouds_per_combination
        self.augment_with_random_points = augment_with_random_points
        self.ds_path = Path(data_dir, phase)
        if not self.ds_path.is_dir():
            raise Exception(f"Could not find a dataset in path [{self.ds_path}]")

        if scanobjectnn:
            random_pc_dir = self.ds_path.joinpath(self.point_cloud_random_dir_name)
            # [:-2] removes the _0 so that when it is added later it will match the file name
            self.file_names = [f.stem[:-2] for f in random_pc_dir.glob("*.npy")]
            self.num_files = len(self.file_names)
            self.size = self.num_files * self.num_point_clouds_per_combination
            return
        print(f"Processing dataset [{phase}] with farthest point sampling...")
        if not self.random_pc:
            generate_point_clouds(data_dir, phase, num_points, self.num_point_clouds_per_combination,
                                  self.point_cloud_fps_dir_name, sampling_method=farthest_point_sampling, gaussian=self.gaussian,
                                  apply_point_cloud_normalization=self.apply_point_cloud_normalization)
        else:
            num_points = self.random_pc
            print(f"Using uniform sampling only with [{num_points}] samples")
        normalize_labels(data_dir, phase, self.yml_gt_normalized_dir_name, params_descriptors, self.train_with_visibility_label)
        print(f"Processing dataset [{phase}] with uniform sampling (augmentation)...")
        generate_point_clouds(data_dir, phase, num_points, self.num_point_clouds_per_combination,
                              self.point_cloud_random_dir_name, sampling_method=sample_surface, gaussian=self.gaussian,
                              apply_point_cloud_normalization=self.apply_point_cloud_normalization)

        obj_gt_dir = self.ds_path.joinpath('obj_gt')
        self.file_names = [f.stem for f in obj_gt_dir.glob("*.obj")]

        self.num_files = len(self.file_names)
        self.size = self.num_files * self.num_point_clouds_per_combination


    def __getitem__(self, _index):
        file_idx = _index // self.num_point_clouds_per_combination
        sample_idx = _index % self.num_point_clouds_per_combination
        file_name = self.file_names[file_idx]

        pc = []
        random_pc_path = self.ds_path.joinpath(self.point_cloud_random_dir_name, f"{file_name}_{sample_idx}.npy")
        fps_pc_path = self.ds_path.joinpath(self.point_cloud_fps_dir_name, f"{file_name}_{sample_idx}.npy")
        if self.random_pc:
            pc = np.load(str(random_pc_path))
            pc = torch.from_numpy(pc).float()
            assert len(pc) == self.random_pc
        else:
            if fps_pc_path.is_file():
                pc = np.load(str(fps_pc_path))
                pc = torch.from_numpy(pc).float()

                # augment the farthest point sampled point cloud with points from a randomly sampled point cloud
                # note that in some tests we did not apply the augmentation
                if self.augment_with_random_points:
                    pc_aug = np.load(str(random_pc_path))
                    pc_aug = torch.from_numpy(pc_aug).float()
                    pc_aug = pc_aug[np.random.choice(pc_aug.shape[0], replace=False, size=800)]
                    pc = torch.cat((pc, pc_aug), dim=0)
            else:
                assert self.phase == "real"

        # assert that the point cloud is normalized
        max_diff = 0.05
        if self.random_pc:
            max_diff = 0.3
        if not self.gaussian or self.gaussian == 0.0:
            max_dist_from_center = abs(1.0 - torch.max(torch.sqrt(torch.sum((pc ** 2), dim=1))))
            assert max_dist_from_center < max_diff, f"Point cloud is not normalized [{max_dist_from_center} > {max_diff}] for sample [{file_name}]. If this is an external ds, please consider using prepare_coseg.py script first."

        # load target vectors, for test phase, some examples may not have a yml file attached to the
        yml_path = self.ds_path.joinpath(self.yml_gt_normalized_dir_name, f"{file_name}.yml")
        yml_obj = None
        if yml_path.is_file():
            with open(yml_path, 'r') as f:
                yml_obj = yaml.load(f, Loader=yaml.FullLoader)
        else:
            # for training and validation we must have a yml file for each sample, for certain phases, yml file is not mandatory
            assert self.phase == "coseg" or self.phase == "real"

        # assemble the vectors in the requested order of parameters
        targets = assemble_targets(yml_obj, self.inputs_to_eval)

        # dataloaders are not allowed to return None, anything empty is converted to []
        return file_name, pc, targets, yml_obj if yml_obj else []

    def __len__(self):
        return self.size
