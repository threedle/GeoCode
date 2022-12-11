import numpy as np
import torch.utils.data as data
import yaml
from pathlib import Path
from .data_processing import normalize_labels
from torchvision import transforms
from PIL import Image
from skimage.morphology import erosion, dilation
import random
from .dataset_util import assemble_targets


class DatasetSketch(data.Dataset):
    def __init__(self,
                 inputs_to_eval,
                 params_descriptors,
                 camera_angles_to_process,
                 pretrained_vgg,
                 data_dir,
                 phase,
                 train_with_visibility_label=True):
        self.inputs_to_eval = inputs_to_eval
        self.data_dir = data_dir
        self.phase = phase
        self.pretrained_vgg = pretrained_vgg
        self.train_with_visibility_label = train_with_visibility_label
        self.camera_angles_to_process = camera_angles_to_process
        self.num_sketches_camera_angles = len(self.camera_angles_to_process)
        self.yml_gt_normalized_dir_name = 'yml_gt_normalized'
        self.ds_path = Path(data_dir, phase)
        if not self.ds_path.is_dir():
            raise Exception(f"Could not find a dataset in path [{self.ds_path}]")
        self.sketches_path = self.ds_path.joinpath("sketches")
        if not self.sketches_path.is_dir():
            raise Exception(f"Could not find a sketches in path [{self.sketches_path}]")
        self.sketch_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        normalize_labels(data_dir, phase, self.yml_gt_normalized_dir_name, params_descriptors, self.train_with_visibility_label)

        obj_gt_dir = self.ds_path.joinpath('obj_gt')
        self.file_names = [f.stem for f in obj_gt_dir.glob("*.obj")]
        if self.phase == "real" or self.phase == "clipasso" or self.phase == "traced":
            self.file_names = [f.stem for f in self.sketches_path.glob("*.png")]

        num_files = len(self.file_names)
        if self.phase == "real" or self.phase == "clipasso" or self.phase == "traced":
            self.size = num_files
        else:
            self.size = num_files * self.num_sketches_camera_angles

    def __getitem__(self, _index):
        if self.phase == "real" or self.phase == "clipasso" or self.phase == "traced":
            file_idx = _index
            sketch_idx = 0
        else:
            file_idx = _index // self.num_sketches_camera_angles
            sketch_idx = _index % self.num_sketches_camera_angles
        file_name = self.file_names[file_idx]

        # load target vectors, for test phase, some examples may not have a yml file attached to them
        yml_path = self.ds_path.joinpath(self.yml_gt_normalized_dir_name, f"{file_name}.yml")
        yml_obj = None
        if yml_path.is_file():
            with open(yml_path, 'r') as f:
                yml_obj = yaml.load(f, Loader=yaml.FullLoader)
        else:
            # for training and validation we must have a yml file for each sample, for certain phases, yml file is not mandatory
            assert self.phase == "test" or self.phase == "coseg" or self.phase == "real" or self.phase == "clipasso" or self.phase == "traced"

        # assemble the vectors in the requested order of parameters
        targets = assemble_targets(yml_obj, self.inputs_to_eval)

        sketch_files = sorted(self.sketches_path.glob(f"{file_name}_*.png"))
        if self.phase == "real" or self.phase == "clipasso" or self.phase == "traced":
            sketch_files = sorted(self.sketches_path.glob(f"{file_name}.png"))
        # filter out sketches from camera angles that are excluded
        if self.phase != "real" and self.phase != "clipasso" and self.phase != "traced":
            sketch_files = [f for f in sketch_files if any( camera_angle in f.name for camera_angle in self.camera_angles_to_process )]
            if len(sketch_files) != len(self.camera_angles_to_process):
                raise Exception(f"Object [{file_name}] is missing sketch files")
        sketch_file = sketch_files[sketch_idx]
        sketch = Image.open(sketch_file).convert("RGB")
        if sketch.size[0] != sketch.size[0]:
            raise Exception(f"Images should be square, got [{sketch.size}] instead.")
        if sketch.size[0] != 224:
            sketch = sketch.resize((224, 224), Image.BILINEAR)
        # augmentation for the sketches
        if self.phase == "train":
            # three augmentation options: 1) original 2) erosion 3) erosion then dilation
            aug_idx = random.randint(0, 2)
            if aug_idx == 1:
                sketch = np.array(sketch)
                sketch = erosion(sketch)
                sketch = Image.fromarray(sketch)
            if aug_idx == 2:
                sketch = np.array(sketch)
                eroded = erosion(sketch)
                sketch = dilation(eroded)
                sketch = Image.fromarray(sketch)
        sketch = self.sketch_transforms(sketch)
        if not self.pretrained_vgg:
            sketch = sketch[0].unsqueeze(0)  # sketch.shape = [1, 224, 224]

        curr_file_camera_angle = 'angle_na'
        for camera_angle in self.camera_angles_to_process:
            if camera_angle in str(sketch_file):
                curr_file_camera_angle = camera_angle
                break
        if self.phase != "real" and self.phase != "clipasso" and self.phase != "traced":
            assert curr_file_camera_angle != 'angle_na'

        # dataloaders are not allowed to return None, anything empty is converted to []
        return file_name, curr_file_camera_angle, sketch, targets, yml_obj if yml_obj else []

    def __len__(self):
        return self.size
