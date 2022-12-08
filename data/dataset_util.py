import torch
import numpy as np
from typing import Dict, List


def assemble_targets(yml_obj: Dict, inputs_to_eval: List[str]):
    targets = []
    if yml_obj:
        for param_name in inputs_to_eval:
            if param_name[-2:] == ' x':
                targets.append(yml_obj[param_name[:-2]]['x'])
            elif param_name[-2:] == ' y':
                targets.append(yml_obj[param_name[:-2]]['y'])
            elif param_name[-2:] == ' z':
                targets.append(yml_obj[param_name[:-2]]['z'])
            else:
                targets.append(yml_obj[param_name])

    # convert from list to numpy array and then to torch tensor
    targets = torch.from_numpy(np.asarray(targets))
    return targets
