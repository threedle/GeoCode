import yaml
import json
import shutil
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from barplot_util import gen_and_save_barplot
from neptune.new.types import File
from models.dgcnn import DGCNN
from models.vgg import vgg11_bn
from models.decoder import DecodersNet, DecodersNetAlex
from calculator_accuracy import AccuracyCalculator
from calculator_loss import LossCalculator
from common.param_descriptors import ParamDescriptors
from pathlib import Path
from geocode_util import InputType
import timeit
import torch.nn as nn
import random


class Model(pl.LightningModule):
    def __init__(self, top_k_acc, batch_size, detailed_vec_size, increase_network_size, normalize_embeddings,
                 pretrained_vgg, input_type, inputs_to_eval, lr, sched_step_size, sched_gamma,
                 exp_name=None, trainer=None, param_descriptors: ParamDescriptors = None, models_dir: Path = None,
                 results_dir: Path = None, test_dir: Path = None, test_dataloaders_types=None, test_input_type=None,
                 use_regression=False, use_resnet=False, discrete=False, continuous=False):
        super().__init__()
        # saved hyper parameters
        self.input_type = input_type
        self.inputs_to_eval = inputs_to_eval
        self.batch_size = batch_size
        self.lr = lr
        self.sched_step_size = sched_step_size
        self.sched_gamma = sched_gamma
        self.top_k_acc = top_k_acc
        self.discrete = discrete
        self.continuous = continuous

        # non-saved parameters
        self.trainer = trainer
        self.param_descriptors = param_descriptors
        self.param_descriptors_map = self.param_descriptors.get_param_descriptors_map()
        self.results_dir = results_dir
        self.test_dir = test_dir
        self.exp_name = exp_name
        self.models_dir = models_dir
        self.test_dataloaders_types = test_dataloaders_types
        self.test_type = test_input_type
        self.use_regression = use_regression
        self.use_resnet = use_resnet

        assert (not discrete and not continuous) or (discrete ^ continuous)
        if discrete:
            self.params_indices = [i for i, param_name in enumerate(self.inputs_to_eval) if self.param_descriptors_map[param_name].input_type in ['Boolean', 'Integer']]
            self.inputs_to_eval = [param_name for param_name in self.inputs_to_eval if self.param_descriptors_map[param_name].input_type in ['Boolean', 'Integer']]
            print("Discrete params only:")
            print(self.inputs_to_eval)
        elif continuous:
            self.params_indices = [i for i, param_name in enumerate(self.inputs_to_eval) if self.param_descriptors_map[param_name].input_type in ['Float', 'Vector']]
            self.inputs_to_eval = [param_name for param_name in self.inputs_to_eval if self.param_descriptors_map[param_name].input_type in ['Float', 'Vector']]
            print("Continuous params only:")
            print(self.inputs_to_eval)

        self.num_inferred_samples = 0
        self.inference_time = 0.0

        regression_params = None
        if self.use_regression:
            regression_params = [param_descriptor.input_type == "Float" or param_descriptor.input_type == "Vector" for
                                 param_name, param_descriptor in self.param_descriptors_map.items()]
        self.acc_calc = AccuracyCalculator(self.inputs_to_eval, self.param_descriptors_map)
        self.loss_calc = LossCalculator(self.inputs_to_eval, self.param_descriptors_map)
        self.decoders_net = DecodersNet(output_channels=detailed_vec_size, increase_network_size=increase_network_size, regression_params=regression_params)

        self.dgcnn = None
        self.vgg = None
        if InputType.pc in self.input_type:
            self.dgcnn = DGCNN(increase_network_size=increase_network_size, normalize_embeddings=normalize_embeddings)
        if InputType.sketch in self.input_type:
            if self.use_resnet:
                self.vgg = timm.create_model('resnet50d', pretrained=pretrained_vgg, in_chans=1)
                self.vgg.fc = nn.Linear(self.vgg.get_classifier().in_features, 128)
            elif self.discrete:
                print("SRPM Discrete")
                self.vgg = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)  # https://pytorch.org/hub/pytorch_vision_alexnet/
                self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # grayscale images, switch to 1 channel input
                del self.vgg.classifier[-1]
                num_classes_list = []
                for param_name in self.inputs_to_eval:
                    if self.param_descriptors_map[param_name].input_type in ['Boolean', 'Integer']:
                        num_classes_list.append(self.param_descriptors_map[param_name].num_classes)
                self.decoders_net = DecodersNetAlex(num_classes_list)  # overriding previous decoder net
            elif self.continuous:
                print("SRPM Continuous")
                self.vgg = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)  # https://pytorch.org/hub/pytorch_vision_alexnet/
                self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)  # grayscale images, switch to 1 channel input
                self.vgg.classifier[-1] = nn.Linear(4096, len(self.inputs_to_eval))
                # at this point the classifier is 9216->4096->4096->45 (this is actually an additional 9216 layer compared to SRPM paper)
            else:
                # default sketch model
                self.vgg = vgg11_bn(pretrained=pretrained_vgg, progress=True, encoder_only=True,
                                    increase_network_size=increase_network_size, normalize_embeddings=normalize_embeddings)
        self.vgg = self.vgg.float()
        self.save_hyperparameters(
            ignore=["trainer", "param_descriptors", "models_dir", "results_dir", "test_dir", "test_input_type",
                    "use_regression", "exp_name", "test_dataloaders_types", "discrete", "continuous"])

    def configure_optimizers(self):
        params = list(self.decoders_net.parameters())
        if InputType.pc in self.input_type:
            params += list(self.dgcnn.parameters())
        if InputType.sketch in self.input_type:
            params += list(self.vgg.parameters())
        optimizer = optim.Adam(params, lr=self.lr)
        lr_scheduler = StepLR(optimizer, step_size=self.sched_step_size, gamma=self.sched_gamma)
        return [optimizer], [lr_scheduler]

    def log_accuracy(self, phase, correct_arr, metric_type):
        """
        :param phase: e.g. "train" or "val"
        :param correct_arr: two dim array, rows are k in top-k accuracy, cols are the parameters
                            the value is the number of correct predictions for a parameter when considering
                            top-k accuracy (top-k with k=1 is equivalent to argmax)
        :param metric_type: either "pc" or "sketch"
        """
        for i, param_name in enumerate(self.inputs_to_eval):
            for j in range(self.top_k_acc):
                acc_metric = correct_arr[j][i] / self.batch_size
                self.log(f"{phase}/acc_top{j + 1}/{metric_type}/{param_name}", acc_metric, on_step=False, on_epoch=True,
                         logger=True, batch_size=self.batch_size)
        for j in range(self.top_k_acc):
            acc_avg_metric = sum(correct_arr[j]) / (self.batch_size * len(self.inputs_to_eval))
            self.log(f"{phase}/acc_top{j + 1}/{metric_type}/avg", acc_avg_metric, on_step=False, on_epoch=True,
                     logger=True, batch_size=self.batch_size)

    def get_decoder_loss(self, pc_emb, sketch_emb, targets_pcs, targets_sketches):
        if InputType.pc in self.input_type:
            batch_size = pc_emb.shape[0]
        else:
            batch_size = sketch_emb.shape[0]
        pred_pc = None
        pred_sketch = None
        if InputType.pc in self.input_type and InputType.sketch in self.input_type:
            targets_both = torch.cat((targets_pcs, targets_sketches), dim=0)
            pred_both = self.decoders_net.decode(torch.cat((pc_emb, sketch_emb), dim=0))
            pred_pc = pred_both[:batch_size, :]
            pred_sketch = pred_both[batch_size:, :]
            decoders_loss, detailed_decoder_loss = self.loss_calc.loss(pred_both, targets_both)
        elif InputType.pc in self.input_type:
            pred_pc = self.decoders_net.decode(pc_emb)
            decoders_loss, detailed_decoder_loss = self.loss_calc.loss(pred_pc, targets_pcs)
        elif InputType.sketch in self.input_type:
            if self.continuous:
                pred_sketch = sketch_emb  # for continuous params, the output from AlexNet is the prediction
                mse_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='none')
                loss = mse_loss(pred_sketch, targets_sketches)
                assert loss.dtype == torch.float
                isr = False  # random.random() < 1
                if isr:
                    print(loss)
                mask = (targets_sketches == -1.0).clone().detach().float()
                if isr:
                    print(mask)
                loss *= (1.0 - mask)
                if isr:
                    print(loss)
                    print(torch.sum(loss))
                    print("---")
                loss = torch.sum(loss, dim=0)
                decoders_loss, detailed_decoder_loss = torch.sum(loss), loss
                assert decoders_loss.dtype == torch.float
                assert detailed_decoder_loss.dtype == torch.float
            elif self.discrete:
                pred_sketch = self.decoders_net.decode(sketch_emb)
                decoders_loss, detailed_decoder_loss = self.loss_calc.loss_discrete(pred_sketch, targets_sketches)
            else:
                pred_sketch = self.decoders_net.decode(sketch_emb)
                decoders_loss, detailed_decoder_loss = self.loss_calc.loss(pred_sketch, targets_sketches)
        else:
            raise Exception("Illegal input type")
        return decoders_loss, detailed_decoder_loss, pred_pc, pred_sketch

    def training_step(self, train_batch, batch_idx):
        targets_pcs = None
        targets_sketches = None
        pc_emb = None
        sketch_emb = None
        if InputType.pc in self.input_type:
            _, pcs, targets_pcs, _ = train_batch["pc"]
            pcs = pcs.transpose(2, 1)
            pc_emb = self.dgcnn(pcs)
        if InputType.sketch in self.input_type:
            _, _, sketches, targets_sketches, _ = train_batch["sketch"]
            sketches = sketches.float()
            targets_sketches = targets_sketches.float()
            if self.discrete or self.continuous:
                targets_sketches = targets_sketches[:, self.params_indices]
            sketch_emb = self.vgg(sketches)

        decoders_loss, detailed_decoder_loss, pred_pc, pred_sketch = self.get_decoder_loss(pc_emb, sketch_emb,
                                                                                           targets_pcs,
                                                                                           targets_sketches)

        # log the current LR
        # Fetch the learning rate from the first optimizer
        lr = None
        for opt in self.trainer.optimizers:
            lr = opt.param_groups[0]['lr']  # Access the learning rate of the first param group
            break  # Stop after the first optimizer
        if lr is not None:
            self.log('train/lr', lr, on_step=True, on_epoch=False, prog_bar=True)

        # log detailed decoding loss
        for i, param_name in enumerate(self.inputs_to_eval):
            self.log(f"train/loss/{param_name}", detailed_decoder_loss[i], on_step=False, on_epoch=True, logger=True,
                     batch_size=self.batch_size)

        train_loss = decoders_loss
        assert train_loss.dtype == torch.float
        self.log("train/loss/total", train_loss, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)

        # compute and log accuracy for point cloud and sketch
        if InputType.pc in self.input_type:
            correct_arr_pc = self.acc_calc.eval(pred_pc, targets_pcs, self.top_k_acc)
            self.log_accuracy("train", correct_arr_pc, "pc")
        if InputType.sketch in self.input_type:
            if self.discrete:
                correct_arr_sketch = self.acc_calc.eval_discrete_only(pred_sketch, targets_sketches, self.top_k_acc)
            elif self.continuous:
                correct_arr_sketch = self.acc_calc.eval_continuous_only(pred_sketch, targets_sketches, self.top_k_acc)
            else:
                correct_arr_sketch = self.acc_calc.eval(pred_sketch, targets_sketches, self.top_k_acc)
            self.log_accuracy("train", correct_arr_sketch, "sketch")

        assert train_loss.dtype == torch.float
        return train_loss

    def validation_step(self, val_batch, batch_idx):
        targets_pcs = None
        targets_sketches = None
        pc_emb = None
        sketch_emb = None
        if InputType.pc in self.input_type:
            _, pcs, targets_pcs, _ = val_batch["pc"]
            pcs = pcs.transpose(2, 1)
            pc_emb = self.dgcnn(pcs)
        if InputType.sketch in self.input_type:
            _, _, sketches, targets_sketches, _ = val_batch["sketch"]
            sketches = sketches.float()
            targets_sketches = targets_sketches.float()
            if self.discrete or self.continuous:
                targets_sketches = targets_sketches[:, self.params_indices]
            sketch_emb = self.vgg(sketches)

        decoders_loss, detailed_decoder_loss, pred_pc, pred_sketch = self.get_decoder_loss(pc_emb, sketch_emb,
                                                                                           targets_pcs,
                                                                                           targets_sketches)
        # log detailed decoding loss
        for i, param_name in enumerate(self.inputs_to_eval):
            self.log(f"val/loss/{param_name}", detailed_decoder_loss[i], on_step=False, on_epoch=True, logger=True,
                     batch_size=self.batch_size)
        val_loss = decoders_loss
        self.log("val/loss/total", val_loss, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)

        assert val_loss.dtype == torch.float
        # compute and log accuracy for point cloud and sketch
        correct_arr_pc = None
        if InputType.pc in self.input_type:
            correct_arr_pc = self.acc_calc.eval(pred_pc, targets_pcs, self.top_k_acc)
            self.log_accuracy("val", correct_arr_pc, "pc")
        correct_arr_sketch = None
        if InputType.sketch in self.input_type:
            if self.discrete:
                correct_arr_sketch = self.acc_calc.eval_discrete_only(pred_sketch, targets_sketches, self.top_k_acc)
            elif self.continuous:
                correct_arr_sketch = self.acc_calc.eval_continuous_only(pred_sketch, targets_sketches, self.top_k_acc)
            else:
                correct_arr_sketch = self.acc_calc.eval(pred_sketch, targets_sketches, self.top_k_acc)
            self.log_accuracy("val", correct_arr_sketch, "sketch")

        # log average validation accuracy
        pc_acc_avg = []
        if InputType.pc in self.input_type:
            for j in range(self.top_k_acc):
                curr_avg = sum(correct_arr_pc[j]) / (self.batch_size * len(self.inputs_to_eval))
                pc_acc_avg.append(curr_avg)
        sketch_acc_avg = []
        if InputType.sketch in self.input_type:
            for j in range(self.top_k_acc):
                curr_avg = sum(correct_arr_sketch[j]) / (self.batch_size * len(self.inputs_to_eval))
                sketch_acc_avg.append(curr_avg)
        avg_acc = []
        for j in range(self.top_k_acc):
            if InputType.pc in self.input_type and InputType.sketch in self.input_type:
                curr_avg = (pc_acc_avg[j] + sketch_acc_avg[j]) / 2
            elif InputType.pc in self.input_type:
                curr_avg = pc_acc_avg[j]
            elif InputType.sketch in self.input_type:
                curr_avg = sketch_acc_avg[j]
            else:
                raise Exception("Illegal input type")
            avg_acc.append(curr_avg)
            self.log(f"val/acc_top{j + 1}/avg", curr_avg, on_step=False, on_epoch=True, logger=True,
                     batch_size=self.batch_size)
        return avg_acc, correct_arr_pc, correct_arr_sketch

    def validation_epoch_end(self, validation_step_outputs):
        num_batches = len(validation_step_outputs)
        num_samples = num_batches * self.batch_size
        for j in range(self.top_k_acc):
            avg_acc = 0
            correct_arr_pc = [0] * len(self.inputs_to_eval)
            correct_arr_sketch = [0] * len(self.inputs_to_eval)
            for avg_acc_batch, correct_arr_pc_batch, correct_arr_sketch_batch in validation_step_outputs:
                avg_acc += avg_acc_batch[j]
                for i in range(len(self.inputs_to_eval)):
                    if correct_arr_pc_batch:
                        correct_arr_pc[i] += correct_arr_pc_batch[j][i]
                    if correct_arr_sketch_batch:
                        correct_arr_sketch[i] += correct_arr_sketch_batch[j][i]
            avg_acc /= len(validation_step_outputs)

            if f'val/acc_top{j + 1}/best_avg' in self.trainer.callback_metrics:
                best_avg_val_acc = max(self.trainer.callback_metrics[f'val/acc_top{j + 1}/best_avg'], avg_acc)
            else:
                best_avg_val_acc = avg_acc
            self.log(f"val/acc_top{j + 1}/best_avg", best_avg_val_acc, on_step=False, on_epoch=True, logger=True,
                     batch_size=self.batch_size)
            if avg_acc == best_avg_val_acc:
                barplot_data = {
                    "inputs_to_eval": self.inputs_to_eval,
                    "correct_arr_pc": correct_arr_pc,
                    "total_pc": num_samples,
                    "correct_arr_sketch": correct_arr_sketch,
                    "total_sketch": num_samples,
                    "accuracy_top_k": j + 1,
                }
                barplot_data_file_path = f'{self.models_dir}/{self.exp_name}/val_barplot_top_{j + 1}.json'
                with open(barplot_data_file_path, 'w') as barplot_data_file:
                    json.dump(barplot_data, barplot_data_file)
                # log the bar plot as image
                fig = gen_and_save_barplot(barplot_data_file_path,
                                           title=f"Validation Accuracy Top {j + 1} @ Epoch {self.trainer.current_epoch}")
                if self.logger:
                    self.logger.run["barplot"].log(File.as_image(fig))

    def _get_huang_continuous_path(self, yaml_file_path, folder_name):
        continuous_yaml_file_path = yaml_file_path.parents[2] / f"results_{self.exp_name.replace('_discrete_', '_cont_')}" / folder_name / yaml_file_path.name
        if not continuous_yaml_file_path.is_file():
            continuous_yaml_file_path = yaml_file_path.parents[2] / f"results_{self.exp_name.replace('_discrete_', '_continuous_')}" / folder_name / yaml_file_path.name
        if not continuous_yaml_file_path.is_file():
            raise Exception(f"Failed when searching the continuous Huang experiment folder [{continuous_yaml_file_path}].")
        return continuous_yaml_file_path

    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        assert self.batch_size == 1
        pc, sketch = None, None
        if self.test_dataloaders_types[dataloader_idx] == 'pc':
            file_name, pc, targets, shape = test_batch
        elif self.test_dataloaders_types[dataloader_idx] == 'sketch':
            file_name, sketch_camera_angle, sketch, targets, shape = test_batch
            if shape and (self.discrete or self.continuous):
                # the existence of "shape" means we have ground truth target values
                targets = targets.float()
                targets = targets[:, self.params_indices]
        else:
            raise Exception(f"Unrecognized dataloader type [{self.test_dataloaders_types[dataloader_idx]}]")

        t_0 = timeit.default_timer()
        if pc is not None:
            pcs = pc.transpose(2, 1)
            pred_pc = self.decoders_net.decode(self.dgcnn(pcs))
            pred_map_pc = self.param_descriptors.convert_prediction_vector_to_map(pred_pc.cpu(), use_regression=self.use_regression)
            self.num_inferred_samples += pred_pc.shape[0]
            self.inference_time += timeit.default_timer() - t_0
            pc_pred_yaml_file_path = self.results_dir.joinpath('yml_predictions_pc', f'{file_name[0]}_pred_pc.yml')
            with open(pc_pred_yaml_file_path, 'w') as yaml_file:
                yaml.dump(pred_map_pc, yaml_file)
        elif sketch is not None:
            sketch_pred_yaml_file_path = self.results_dir.joinpath('yml_predictions_sketch', f'{file_name[0]}_{sketch_camera_angle[0]}_pred_sketch.yml')
            if self.continuous:
                pred_sketch = self.vgg(sketch)  # no decoding is needed
                pred_map_sketch = self.param_descriptors.convert_prediction_vector_to_map_continuous_only(self.inputs_to_eval, pred_sketch.cpu())
            elif self.discrete:
                pred_sketch = self.decoders_net.decode(self.vgg(sketch))
                pred_map_sketch = self.param_descriptors.convert_prediction_vector_to_map_discrete_only(self.inputs_to_eval, pred_sketch.cpu())
                continuous_yaml_file_path = self._get_huang_continuous_path(sketch_pred_yaml_file_path, "yml_predictions_sketch")
                with open(continuous_yaml_file_path, 'r') as continuous_yaml_file:
                    pred_map_sketch_continuous = yaml.load(continuous_yaml_file, Loader=yaml.FullLoader)
                pred_map_sketch.update(pred_map_sketch_continuous)
            else:
                pred_sketch = self.decoders_net.decode(self.vgg(sketch))
                pred_map_sketch = self.param_descriptors.convert_prediction_vector_to_map(pred_sketch.cpu(), use_regression=self.use_regression)
            self.num_inferred_samples += pred_sketch.shape[0]
            self.inference_time += timeit.default_timer() - t_0
            with open(sketch_pred_yaml_file_path, 'w') as yaml_file:
                yaml.dump(pred_map_sketch, yaml_file)
        else:
            raise Exception("No pc and no sketch input")

        # compute accuracy for point cloud and sketch
        correct_arr_pc = None
        correct_arr_sketch = None
        if shape:
            # this means we had a target vector to compare against
            if pc is not None:
                correct_arr_pc = self.acc_calc.eval(pred_pc, targets, self.top_k_acc)
                expanded_targets_vector = self.param_descriptors.expand_target_vector(targets.cpu())
                gt_map = self.param_descriptors.convert_prediction_vector_to_map(expanded_targets_vector, use_regression=self.use_regression)
            if sketch is not None:
                gt_yaml_file_path = self.results_dir.joinpath('yml_gt', f'{file_name[0]}_gt.yml')
                if self.discrete:
                    correct_arr_sketch = self.acc_calc.eval_discrete_only(pred_sketch, targets, self.top_k_acc)
                    expanded_targets_vector = self.param_descriptors.expand_target_vector_discrete_only(self.inputs_to_eval, targets.cpu())
                    gt_map = self.param_descriptors.convert_prediction_vector_to_map_discrete_only(self.inputs_to_eval, expanded_targets_vector)
                    continuous_yaml_file_path = self._get_huang_continuous_path(gt_yaml_file_path, "yml_gt")
                    with open(continuous_yaml_file_path, 'r') as continuous_yaml_file:
                        gt_map_continuous = yaml.load(continuous_yaml_file, Loader=yaml.FullLoader)
                    gt_map.update(gt_map_continuous)
                elif self.continuous:
                    correct_arr_sketch = self.acc_calc.eval_continuous_only(pred_sketch, targets, self.top_k_acc)
                    expanded_targets_vector = self.param_descriptors.expand_target_vector_contiuous_only(self.inputs_to_eval, targets.cpu())
                    gt_map = self.param_descriptors.convert_prediction_vector_to_map_continuous_only(self.inputs_to_eval, expanded_targets_vector)
                else:
                    correct_arr_sketch = self.acc_calc.eval(pred_sketch, targets, self.top_k_acc)
                    expanded_targets_vector = self.param_descriptors.expand_target_vector(targets.cpu())
                    gt_map = self.param_descriptors.convert_prediction_vector_to_map(expanded_targets_vector, use_regression=self.use_regression)
            # save ground truth yaml
            with open(gt_yaml_file_path, 'w') as yaml_file:
                yaml.dump(gt_map, yaml_file)

        # save the gt sketches, note that obj gt are saved during the visualization step
        sketches_dir = self.test_dir.joinpath("sketches")
        if sketches_dir.is_dir():
            sketch_files = sketches_dir.glob(f'{file_name[0]}*')
            for sketch_file in sketch_files:
                shutil.copy(sketch_file, self.results_dir.joinpath('sketch_gt', sketch_file.name))

        return correct_arr_pc, correct_arr_sketch

    def test_epoch_end(self, test_step_outputs):
        assert self.batch_size == 1
        assert self.test_type
        if InputType.pc in self.test_type and InputType.sketch in self.test_type:
            test_step_outputs_pc_and_sketch = test_step_outputs[0] + test_step_outputs[1]
        else:
            test_step_outputs_pc_and_sketch = test_step_outputs
        for top_k in range(self.top_k_acc):
            total_pcs = 0
            total_sketches = 0
            correct_arr_pc = [0] * len(self.inputs_to_eval)
            correct_arr_sketch = [0] * len(self.inputs_to_eval)
            for correct_arr_pc_batch, correct_arr_sketch_batch in test_step_outputs_pc_and_sketch:
                if correct_arr_pc_batch is not None:
                    total_pcs += 1
                    for i in range(len(self.inputs_to_eval)):
                        correct_arr_pc[i] += correct_arr_pc_batch[top_k][i]
                if correct_arr_sketch_batch is not None:
                    total_sketches += 1
                    for i in range(len(self.inputs_to_eval)):
                        correct_arr_sketch[i] += correct_arr_sketch_batch[top_k][i]
            barplot_data = {
                "inputs_to_eval": self.inputs_to_eval,
                "correct_arr_pc": correct_arr_pc,
                "total_pc": total_pcs,
                "correct_arr_sketch": correct_arr_sketch,
                "total_sketch": total_sketches,
                "accuracy_top_k": top_k + 1,
            }
            if total_pcs > 0 and total_sketches > 0:
                print(
                    f'Saving test barplot for [{total_pcs}] point cloud sampels and [{total_sketches}] sketch samples')
                barplot_data_file_path = f'{self.models_dir}/{self.exp_name}/test_barplot_top_{top_k + 1}.json'
                with open(barplot_data_file_path, 'w') as barplot_data_file:
                    json.dump(barplot_data, barplot_data_file)
