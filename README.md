# GeoCode: Interpretable Shape Programs

*Ofek Pearl, Itai Lang, Yuhua Hu, Raymond A. Yeh, Rana Hanocka*

![alt GeoCode](resources/teaser.png)

> We present GeoCode, a technique for 3D shape synthesis using an intuitively editable parameter space. We build a novel program that enforces a complex set of rules and enables users to perform intuitive and controlled high-level edits that procedurally propagate at a low level to the entire shape. Our program produces high-quality mesh outputs by construction. We use a neural network to map a given point cloud or sketch to our interpretable parameter space. Once produced by our procedural program, shapes can be easily modified. Empirically, we show that GeoCode can infer and recover 3D shapes more accurately compared to existing techniques and we demonstrate its ability to perform controlled local and global shape manipulations.

## Run the test using our dataset and checkpoint

### Installation

Clone and create the Conda environment
```bash
git clone https://github.com/threedle/GeoCode.git
cd GeoCode
conda env create -f environment.yml
conda activate geocode
python setup.py install

# Install Blender 3.2 under `~/Blender`
./scripts/install_blender3.2.sh

# Download the dataset (`~/datasets`), checkpoint (`~/models`) and blend file (`~/blends`) of the `chair` domain
python scripts/download_ds.py --domain chair --datasets-dir ~/datasets --models-dir ~/models --blends-dir ~/blends
```

`vase` and `table` domains are also available

### Run the test for the chair domain

Run the test for the `chair` domain using the downloaded checkpoint, make sure the directories match the directories using in the download step
```bash
cd GeoCode
conda activate geocode
python geocode/geocode.py test --blend-file ~/blends/procedural_chair.blend --models-dir ~/models --dataset-dir ~/datasets --input-type pc sketch --phase test --exp-name exp_geocode_chair
```

This will generate the results in the following directory structure, in 
```
<datasets-dir>
│
└───ChairDataset
    │
    └───test
        │
        └───resutls_<experiment_name>
            │
            └───barplot                    <-- model accuracy graph
            └───obj_gt                     <-- 3D objects of the ground truth samples
            └───obj_predictions_pc         <-- 3D objects predicted from point cloud input
            └───obj_predictions_sketch     <-- 3D objects predicted from sketch input
            └───yml_gt                     <-- labels of the ground truth objects
            └───yml_predictions_pc         <-- labels of the objects predicted from point cloud input
            └───yml_predictions_sketch     <-- labels of the objects predicted from sketch input
```

We also provide a way to automatically render the resulting 3D objects

```bash
cd GeoCode
conda activate geocode
~/Blender/blender-3.2.0-linux-x64/blender ~/blends/procedural_chair.blend -b --python visualize_results/visualize.py -- --dataset-dir ~/datasets --dataset-name ChairDataset --phase test --exp-name exp_geocode_chair
```

this will generate the following additional directories:
```
            ⋮
            └───render_gt                  <-- renders of the ground truth objects
            └───render_predictions_pc      <-- renders of the objects predicted from point cloud input
            └───render_predictions_sketch  <-- renders of the objects predicted from sketch input
```

Example for output

TODO: add images here

## Run training on our dataset

TODO: complete this

## Inspecting the blend files

Open one of the Blend files using Blender 3.2.

To modify the shape using the parameters and to inspect the Geometry Nodes Program click the "Geometry Node" workspace at the top of the window

![alt GeoCode](resources/geo_nodes_button.png)

Then you will see the following screen

![alt GeoCode](resources/geo_nodes_workspace.png)

# Detailed WIKI


## Download the datasets, blend files, and checkpoint files

Download one or more domains:
```bash
python scripts/download_ds.py --domain chair --datasets-dir ~/datasets --models-dir ~/models --blends-dir ~/blends
python scripts/download_ds.py --domain vase --datasets-dir ~/datasets --models-dir ~/models --blends-dir ~/blends
python scripts/download_ds.py --domain table --datasets-dir ~/datasets --models-dir ~/models --blends-dir ~/blends
```

```bash
usage: download_ds [-h] --domain {chair,vase,table} [--datasets-dir DATASETS_DIR] [--models-dir MODELS_DIR] [--blends-dir BLENDS_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --domain {chair,vase,table}
                        The domain name to download the dataset for.
  --datasets-dir DATASETS_DIR
                        The directory to download the dataset to.
  --models-dir MODELS_DIR
                        The directory to download checkpoint file to.
  --blends-dir BLENDS_DIR
                        The directory to download blend file to.
```

Each domain's dataset has the same directory structure, for example, the chair domain command above will create the following directory structure:
```
<datasets-dir>
│
└───ChairDataset
    │
    └───recipe.yml
    │
    └───train
    │   └───obj_gt
    │   └───point_cloud_fps
    │   └───point_cloud_random
    │   └───sketches
    │   └───yml_gt
    │   └───yml_gt_normalized
    │
    └───val
    │   └───obj_gt
    │   └───...
    │
    └───test
        └───obj_gt
        └───...
        
<models-dir>
│
└───exp_geocode_chair
    │
    └───procedural_chair_epoch585_ckpt.zip
    └───procedural_chair_last_ckpt.zip
    └───procedural_chair_epoch585.ckpt
    └───last.ckpt

<blends-dir>
│
└───procedural_chair.blend
```

## Train (1 GPU and 5 cpus setup is recommended)
Training from a checkpoint or new training is done similarly, and only depends on the existence of a `latest` checkpoint file in the experiment directory.
Please note that training using our checkpoints will show start epoch of 0.

```bash
cd GeoCode
conda activate geocode
python geocode/geocode.py train --models-dir ~/models --dataset-dir ~/datasets --nepoch=600 --batch_size=33 --input-type pc sketch --exp-name exp_geocode_chair
```

For logging during training, we encourage the use of [neptune.ai](https://neptune.ai/).
First open an account and create a project, create the file `GeoCode/config/neptune_config.yml` with the following content:

```
neptune:
  api_token: "<TOKEN>"
  project: "<POJECT_PATH>"
```

## Test (1 GPU and 20 cpus setup is recommended)
Testing experiment `experiment_name` on phase `test` of the `chair` domain, will create the directory `resutls_<experiment_name>` with the following directory structure:

```
<datasets-dir>
│
└───ChairDataset
    │
    └───...
    │
    └───test
        └───obj_gt
        └───...
        └───resutls_<experiment_name>
            │
            └───recipe.yml
            │
            └───barplot                    <-- model accuracy graph
            └───obj_gt                     <-- 3D objects of the ground truth samples 
            └───obj_predictions_pc         <-- 3D objects predicted from point cloud input
            └───obj_predictions_sketch     <-- 3D objects predicted from sketch input
            └───render_gt                  <-- renders of the ground truth objects
            └───render_predictions_pc      <-- renders of the objects predicted from point cloud input
            └───render_predictions_sketch  <-- renders of the objects predicted from sketch input
            └───yml_gt                     <-- labels of the ground truth objects
            └───yml_predictions_pc         <-- labels of the objects predicted from point cloud input
            └───yml_predictions_sketch     <-- labels of the objects predicted from sketch input
```

```bash
cd GeoCode
conda activate geocode
python geocode/geocode.py test --blend-file ~/blends/procedural_chair.blend --models-dir ~/models --dataset-dir ~/datasets --input-type pc sketch --phase test --exp-name exp_geocode_chair
```

## Visualize the results (usage of multiple nodes with GPU is recommended)

Visualizing the results. This will render an image for each ground truth and prediction object that were created while running the test script.

```bash
cd GeoCode
conda activate geocode
~/Blender/blender-3.2.0-linux-x64/blender ~/blends/procedural_chair.blend -b --python visualize_results/visualize.py -- --dataset-dir ~/datasets --dataset-name ChairDataset --phase test --exp-name exp_geocode_chair
```

You can also run this in parallel, for example, with 10 processes, by adding the flags `--parallel 10 --mod $NODE_ID`

## Create a new dataset

### Step 1 - define the dataset
Edit (optional) the `dataset_generation` or the `camera_angles` sections in the appropriate `recipe` YAML file. For example, for the chair domain, edit the following recipe file:
`GeoCode/dataset_generator/recipe_files/chair_recipe.yml`

### Step 2 - create the raw objects (20 CPUs with 8GB memory per CPU is recommended)
In this step no GPU and no conda env are required.

For example, generating the val, test, and train datasets for the chair domain, with 3, 3, and 30 shape variation per parameter value, is done using the following commands: 

```bash
cd GeoCode
~/Blender/blender-3.2.0-linux-x64/blender ~/blends/procedural_chair.blend -b --python dataset_generator/dataset_generator.py -- generate-dataset --dataset-dir ~/datasets/MyChairDataset --domain chair --phase val --num-variations 3 --parallel 20
~/Blender/blender-3.2.0-linux-x64/blender ~/blends/procedural_chair.blend -b --python dataset_generator/dataset_generator.py -- generate-dataset --dataset-dir ~/datasets/MyChairDataset --domain chair --phase test --num-variations 3 --parallel 20
~/Blender/blender-3.2.0-linux-x64/blender ~/blends/procedural_chair.blend -b --python dataset_generator/dataset_generator.py -- generate-dataset --dataset-dir ~/datasets/MyChairDataset --domain chair --phase train --num-variations 30 --parallel 20
```

For vase domain
```bash
~/Blender/blender-3.2.0-linux-x64/blender ~/blends/procedural_vase.blend -b --python dataset_generator/dataset_generator.py -- generate-dataset --dataset-dir ~/datasets/MyVaseDataset --domain vase --phase val --num-variations 3 --parallel 20
~/Blender/blender-3.2.0-linux-x64/blender ~/blends/procedural_vase.blend -b --python dataset_generator/dataset_generator.py -- generate-dataset --dataset-dir ~/datasets/MyVaseDataset --domain vase --phase test --num-variations 3 --parallel 20
~/Blender/blender-3.2.0-linux-x64/blender ~/blends/procedural_vase.blend -b --python dataset_generator/dataset_generator.py -- generate-dataset --dataset-dir ~/datasets/MyVaseDataset --domain vase --phase train --num-variations 30 --parallel 20
```

For table dataset
```bash
~/Blender/blender-3.2.0-linux-x64/blender ~/blends/procedural_table.blend -b --python dataset_generator/dataset_generator.py -- generate-dataset --dataset-dir ~/datasets/MyTableDataset --domain table --phase val --num-variations 3 --parallel 20
~/Blender/blender-3.2.0-linux-x64/blender ~/blends/procedural_table.blend -b --python dataset_generator/dataset_generator.py -- generate-dataset --dataset-dir ~/datasets/MyTableDataset --domain table --phase test --num-variations 3 --parallel 20
~/Blender/blender-3.2.0-linux-x64/blender ~/blends/procedural_table.blend -b --python dataset_generator/dataset_generator.py -- generate-dataset --dataset-dir ~/datasets/MyTableDataset --domain table --phase train --num-variations 30 --parallel 20
```

### Step 3 - create the sketches (usage of multiple nodes with GPU is recommended)

This step does not require a conda env but requires GPU(s)

```bash
cd GeoCode
~/Blender/blender-3.2.0-linux-x64/blender ~/blends/procedural_vase.blend -b dataset_generators/sketch_generator.py -- --dataset-dir ~/datasets/MyChairDataset --phases val test train
```

You can also run this in parallel, for example, with 10 processes, by adding the flags `--parallel 10 --mod $NODE_ID`

Once you train on the new dataset, a preprocessing step of the dataset will be performed. The point cloud sampling and normalized label directories and files will be created under the dataset directory. Refer to the `Download the datasets` section at the top of this readme for the directory structure.

## COSEG dataset

Before using GeoCode on COSEG dataset, the samples must be converted from .off to .obj and normalized.
Simply download the dataset from [COSEG's dataset website](http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/ssd.htm), then use the following command:

Example for COSEG's 400 chairs dataset:
```bash
cd ~/datasets/
mkdir CosegChairs400
cd CosegChairs400
wget http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Chairs/shapes.zip
unzip shapes.zip
mkdir coseg
```

then:

```bash
cd GeoCode/dataset_processing
git clone https://github.com/tforgione/model-converter-python.git
~/Blender/blender-3.2.0-linux-x64/blender -b --python prepare_coseg.py -- --shapes-dir ~/datasets/CosegChairs400/shapes --target-dataset-dir ~/datasets/CosegChairs400/ --target-phase coseg
```

## Simplify a dataset (20 CPUs setup is recommended)
Simplifying a mesh is an operation that reduces complexity of the shape's topology.
To simplify the objects in a specific phase in the dataset, use the followingt command:

```bash
cd GeoCode/dataset_processing
conda activate geocode
python simplified_mesh_dataset.py --blend-file ~/blends/geocode.blend --dataset-dir ~/datasets/ChairDataset --src-phase test --dst-phase simplified --simplification-ratios 0.1 0.005
```


## Stability metric (20 CPUs setup is recommended)
To evaluate a tested dataset phase using our *stability metric* use the following command:

```bash
cd GeoCode/stability_metric
python stability_parallel.py --blender-exe ~/Blender/blender-3.2.0-linux-x64/blender --dir-path ~/datasets/ChairDataset/val/obj_gt
```


```bash
usage: stability_parallel [-h] --dir-path DIR_PATH --blender-exe BLENDER_EXE [--skip-components-check] [--apply-normalization]
                          [--limit LIMIT]

options:
  -h, --help            show this help message and exit
  --dir-path DIR_PATH   Path to the dir to test with the 'stability metric'
  --blender-exe BLENDER_EXE
                        Path to Blender executable
  --skip-components-check
                        Skip checking if the shape is structurally valid
  --apply-normalization
                        Apply normalization on the imported objects
  --limit LIMIT         Limit the number of shapes that will be evaluated, randomly selected shapes will be tested
```

## Code structure

- `common` - util packages and classes that are shared by multiple other directories
- `config` - contains the neptune.ai configurations file
- `data` - dataset related classes
- `dataset_processing` - scripts that are intended to manipulate existing datasets
- `geocode` - main training and testing code
- `models` - point cloud and sketch encoders and the decoders network
- `scripts` - contains script to set up our datasets and saved checkpoints 
- `stability_metric` - script to evaluate a tested phase using our *stability metric*
- `visualize_results` - script to generate the renders for all ground truth and predicted shapes
