import torch
from calculator_util import eval_metadata


MSE = torch.nn.MSELoss()
CElossSum = torch.nn.CrossEntropyLoss(reduction='sum')


class LossCalculator():
    def __init__(self, inputs_to_eval, param_descriptors):
        self.inputs_to_eval = inputs_to_eval
        self.normalized_classes_all, self.num_classes_all_shifted_cumulated, self.num_classes_all, self.regression_params_indices \
            = eval_metadata(inputs_to_eval, param_descriptors)
        self.param_descriptors = param_descriptors

    def loss(self, pred, targets):
        """
        _pred: (B, TARGET_VEC_LEN)
        """
        batch_size = pred.shape[0]
        device = targets.device
        normalized_classes_all = self.normalized_classes_all.to(device)
        num_classes_all_shifted_cumulated = self.num_classes_all_shifted_cumulated.to(device)
        num_classes_all = self.num_classes_all.to(device)
        targets_interleaved = torch.repeat_interleave(targets, num_classes_all.view(-1), dim=1)
        normalized_classes_all_repeated = normalized_classes_all.repeat(batch_size, 1).to(device)
        target_class = torch.abs(normalized_classes_all_repeated - targets_interleaved)
        target_class = torch.where(target_class < 1e-3)[1].view(batch_size, -1)  # take the indices along dim=1
        if len(self.regression_params_indices) > 0:
            regression_params_indices_repeated = self.regression_params_indices.repeat(batch_size, 1).to(device)
            target_class = torch.cat((target_class, regression_params_indices_repeated), dim=1)
            target_class, _ = torch.sort(target_class, dim=1)
        assert target_class.shape[1] == len(self.inputs_to_eval)
        target_class = target_class - num_classes_all_shifted_cumulated
        # target_class = target_class.to(_pred.get_device())
        pred_split = torch.split(pred, list(num_classes_all), dim=1)
        detailed_ce_loss = [(CElossSum(p, t) if p.shape[1] > 1 else None) for p, t in zip( pred_split, target_class.T )]

        detailed_mse_loss = [None] * targets.shape[1]
        if len(self.regression_params_indices) > 0:
            for param_idx, (p, t) in enumerate(zip(pred_split, targets.T)):
                if self.param_descriptors[self.inputs_to_eval[param_idx]].is_regression:
                    t_visibility = torch.zeros(t.shape[0])
                    t_visibility[t >= 0.0] = 0.0
                    t_visibility[t == -1.0] = 1.0
                    t_visibility = t_visibility.to(device)
                    t_clone = t.clone()
                    t_clone = t_clone.float()
                    t_clone[t_clone == -1] = p[t_clone == -1,0]
                    t_adjusted = torch.concat((t_clone.unsqueeze(1), t_visibility.unsqueeze(1)), dim=1)
                    detailed_mse_loss[param_idx] = MSE(p, t_adjusted)
        detailed_mse_loss_no_none = [e for e in detailed_mse_loss if e]
        detailed_ce_loss_no_none = [e for e in detailed_ce_loss if e]
        mse_loss_range = 1.0 if not detailed_mse_loss_no_none else (max(detailed_mse_loss_no_none).item() - min(detailed_mse_loss_no_none).item())
        ce_loss_range = max(detailed_ce_loss_no_none).item() - min(detailed_ce_loss_no_none).item()
        detailed_loss = [(ce_loss / ce_loss_range) if not mse_loss else (mse_loss / mse_loss_range) for ce_loss, mse_loss in zip(detailed_ce_loss, detailed_mse_loss)]

        return sum(detailed_loss), detailed_loss
