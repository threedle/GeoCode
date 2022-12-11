import yaml
import hashlib
import numpy as np
from pathlib import Path
from typing import Union


def save_yml(yml_obj, target_yml_file_path):
    with open(target_yml_file_path, 'w') as target_yml_file:
        yaml.dump(yml_obj, target_yml_file, sort_keys=False, width=1000)


def get_source_recipe_file_path(domain):
    """
    get the path to the recipe file path that is found in the source code under the directory "recipe_files"
    """
    return Path(__file__).parent.joinpath('..', 'dataset_generator', 'recipe_files', f'recipe_{domain}.yml').resolve()


def hash_file_name(file_name):
    return int(hashlib.sha1(file_name.encode("utf-8")).hexdigest(), 16) % (10 ** 8)


def get_recipe_yml_obj(recipe_file_path: Union[str, Path]):
    with open(recipe_file_path, 'r') as recipe_file:
        recipe_yml_obj = yaml.load(recipe_file, Loader=yaml.FullLoader)
    return recipe_yml_obj


def load_obj(file: str):
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        split_line = line.split()
        if not split_line:
            continue
        elif split_line[0] == 'v':
            vs.append([float(v) for v in split_line[1:4]])
        elif split_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in split_line[1:]]
            assert len(face_vertex_ids) == 3
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=np.int64)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces
