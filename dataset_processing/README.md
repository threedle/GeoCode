# Additional Dataset-Processing Scripts

## COSEG dataset

Before using GeoCode on COSEG dataset, the samples must be converted from .off to .obj and normalized.
Simply download the dataset from [COSEG's dataset website](http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/ssd.htm):

Example for COSEG's 400 chairs dataset:
```bash
cd ~/datasets/
mkdir CosegChairs400
cd CosegChairs400
wget http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Chairs/shapes.zip
unzip shapes.zip
mkdir coseg
```

then process the dataset using:

```bash
cd GeoCode/dataset_processing
git clone https://github.com/tforgione/model-converter-python.git
~/Blender/blender-3.2.0-linux-x64/blender -b --python prepare_coseg.py -- --shapes-dir ~/datasets/CosegChairs400/shapes --target-dataset-dir ~/datasets/CosegChairs400/ --target-phase coseg
```

for help:

```bash
cd GeoCode/dataset_processing
~/Blender/blender-3.2.0-linux-x64/blender -b --python prepare_coseg.py -- --help
```

## Simplify a dataset (20 CPUs setup is recommended)
Simplifying a mesh is an operation that reduces complexity of the shape's topology.
To simplify the objects in a specific phase in the dataset, use the followingt command:

```bash
cd GeoCode/dataset_processing
conda activate geocode
python simplified_mesh_dataset.py --blend-file ~/blends/geocode.blend --dataset-dir ~/datasets/ChairDataset --src-phase test --dst-phase simplified --simplification-ratios 0.1 0.005
```