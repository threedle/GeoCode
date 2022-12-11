import torch
from calculator_util import eval_metadata


class AccuracyCalculator():
    def __init__(self, inputs_to_eval, param_descriptors):
        self.inputs_to_eval = inputs_to_eval
        self.normalized_classes_all, self.num_classes_all_shifted_cumulated, self.num_classes_all, self.regression_params_indices \
            = eval_metadata(inputs_to_eval, param_descriptors)
        self.param_descriptors = param_descriptors

    def eval(self, pred, targets, top_k_acc):
        batch_size = pred.shape[0]
        device = targets.device
        normalized_classes_all = self.normalized_classes_all.to(device)
        num_classes_all_shifted_cumulated = self.num_classes_all_shifted_cumulated.to(device)
        num_classes_all = self.num_classes_all.to(device)
        correct = [[0] * len(self.inputs_to_eval) for _ in range(top_k_acc)]
        targets_interleaved = torch.repeat_interleave(targets, num_classes_all.view(-1), dim=1)
        normalized_classes_all_repeated = normalized_classes_all.repeat(batch_size, 1).to(device)
        target_class = torch.abs(normalized_classes_all_repeated - targets_interleaved)
        target_class = torch.where(target_class < 1e-3)[1].view(batch_size, -1)  # take the indices along dim=1 since target is of size [1, param_count]
        if len(self.regression_params_indices) > 0:
            regression_params_indices_repeated = self.regression_params_indices.repeat(batch_size, 1).to(device)
            target_class = torch.cat((target_class, regression_params_indices_repeated), dim=1)
            target_class, _ = torch.sort(target_class, dim=1)
        assert target_class.shape[1] == len(self.inputs_to_eval)
        target_class = target_class - num_classes_all_shifted_cumulated
        pred_split = torch.split(pred, list(num_classes_all), dim=1)
        class_indices_diff = [(torch.argmax(p, dim=1) - t if p.shape[1] > 1 else None) for p, t in zip( pred_split, target_class.T )]

        l1_distance = [None] * targets.shape[1]
        if len(self.regression_params_indices) > 0:
            for param_idx, (p, t) in enumerate(zip(pred_split, targets.T)):
                if self.param_descriptors[self.inputs_to_eval[param_idx]].is_regression:
                    adjusted_pred = p[:, 0].clone()
                    adjusted_pred[p[:, 1] >= 0.5] = -1.0
                    l1_distance[param_idx] = torch.abs(adjusted_pred.squeeze() - t)

        for i, param_name in enumerate(self.inputs_to_eval):
            if self.param_descriptors[param_name].is_regression:
                # regression parameter
                normalized_acc_threshold = self.param_descriptors[param_name].normalized_acc_threshold
                for j in range(top_k_acc):
                    assert len(l1_distance[i]) == batch_size
                    correct[j][i] += torch.sum((l1_distance[i] < normalized_acc_threshold * (j + 1)).int()).item()
            else:
                cid = class_indices_diff[i]
                assert len(cid) == batch_size
                for j in range(top_k_acc):
                    correct[j][i] += len(cid[(cid <= j) & (cid >= -j)])
        return correct
