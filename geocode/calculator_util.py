import torch
from typing import Dict, List
from common.param_descriptors import ParamDescriptor


def eval_metadata(inputs_to_eval: List[str], param_descriptors_map: Dict[str, ParamDescriptor]):
    num_classes_all = torch.empty(0)
    normalized_classes_all = torch.empty(0)
    for i, param_name in enumerate(inputs_to_eval):
        param_descriptor = param_descriptors_map[param_name]
        num_classes = param_descriptor.num_classes  # Including the visibility label. If using regression then num_classes=2.
        num_classes_all = torch.cat((num_classes_all, torch. tensor([num_classes]))).long()
        if param_descriptor.normalized_classes is not None:
            normalized_classes = torch.from_numpy(param_descriptor.normalized_classes)
        else:
            # high values so that eval and loss methods will work when using regression
            normalized_classes = torch.tensor([100000.0, 100000.0])
        normalized_classes_all = torch.cat((normalized_classes_all, normalized_classes.view(-1)))
    num_classes_all_shifted = torch.cat((torch.tensor([0]), num_classes_all))[0:-1]  # shift right + drop right-most element
    num_classes_all_shifted_cumulated = torch.cumsum(num_classes_all_shifted, dim=0).view(1, -1)

    # get the indices of all the regression params, then shift them to match the expanded vector
    regression_params = torch.tensor([param_descriptors_map[param_name].is_regression for param_name in inputs_to_eval], dtype=torch.int)
    regression_params_indices = torch.where(regression_params)[0]
    regression_params_indices = torch.tensor([num_classes_all_shifted_cumulated[0, idx] for idx in regression_params_indices])

    return normalized_classes_all, num_classes_all_shifted_cumulated, num_classes_all, regression_params_indices
