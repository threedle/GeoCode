import yaml
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
import neptune.new as neptune
from data.dataset_pc import DatasetPC
from data.dataset_sketch import DatasetSketch
from common.param_descriptors import ParamDescriptors
from common.file_util import get_recipe_yml_obj
from pytorch_lightning.trainer.supporters import CombinedLoader
from geocode_util import InputType, get_inputs_to_eval, calc_prediction_vector_size


def train(opt):
    torch.set_printoptions(precision=4)
    torch.multiprocessing.set_sharing_strategy('file_system')  # to prevent "received 0 items of data" errors
    recipe_file_path = Path(opt.dataset_dir, 'recipe.yml')
    if not recipe_file_path.is_file():
        raise Exception(f'No \'recipe.yml\' file found in path [{recipe_file_path}]')
    recipe_yml_obj = get_recipe_yml_obj(str(recipe_file_path))

    inputs_to_eval = get_inputs_to_eval(recipe_yml_obj)

    top_k_acc = 2
    camera_angles_to_process = [f'{a}_{b}' for a, b in recipe_yml_obj['camera_angles_train']]
    param_descriptors = ParamDescriptors(recipe_yml_obj, inputs_to_eval, use_regression=opt.use_regression, train_with_visibility_label=(not opt.huang))
    param_descriptors_map = param_descriptors.get_param_descriptors_map()
    detailed_vec_size = calc_prediction_vector_size(param_descriptors_map)
    print(f"Prediction vector length is set to [{sum(detailed_vec_size)}]")

    # create datasets
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loaders_map = {}
    val_loaders_map = {}

    num_workers = 5  # use 1 when debugging, otherwise 5

    # pc
    if InputType.pc in opt.input_type:
        train_dataset_pc = DatasetPC(inputs_to_eval, device, param_descriptors_map, opt.dataset_dir, "train", augment_with_random_points=True)
        train_dataloader_pc = DataLoader(train_dataset_pc, batch_size=opt.batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=5)
        val_dataset_pc = DatasetPC(inputs_to_eval, device, param_descriptors_map, opt.dataset_dir, "val", augment_with_random_points=True)
        val_dataloader_pc = DataLoader(val_dataset_pc, batch_size=opt.batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=5)
        train_loaders_map['pc'] = train_dataloader_pc
        val_loaders_map['pc'] = val_dataloader_pc
        print(f"Point cloud train dataset size [{len(train_dataset_pc)}] val dataset size [{len(val_dataset_pc)}]")

    # sketch
    if InputType.sketch in opt.input_type:
        train_dataset_sketch = DatasetSketch(inputs_to_eval, param_descriptors_map, camera_angles_to_process, opt.pretrained_vgg, opt.dataset_dir, "train")
        train_dataloader_sketch = DataLoader(train_dataset_sketch, batch_size=opt.batch_size, shuffle=True, num_workers=num_workers, prefetch_factor=5)
        val_dataset_sketch = DatasetSketch(inputs_to_eval, param_descriptors_map, camera_angles_to_process, opt.pretrained_vgg, opt.dataset_dir, "val")
        val_dataloader_sketch = DataLoader(val_dataset_sketch, batch_size=opt.batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=5)
        train_loaders_map['sketch'] = train_dataloader_sketch
        val_loaders_map['sketch'] = val_dataloader_sketch
        print(f"Sketch train dataset size [{len(train_dataset_sketch)}] val dataset size [{len(val_dataset_sketch)}]")

    combined_train_dataloader = CombinedLoader(train_loaders_map, mode="max_size_cycle")
    combined_val_dataloader = CombinedLoader(val_loaders_map, mode="max_size_cycle")

    if InputType.pc in opt.input_type and InputType.sketch in opt.input_type:
        assert ( len(camera_angles_to_process) * len(train_dataset_pc) ) == len(train_dataset_sketch)
        assert ( len(camera_angles_to_process) * len(val_dataset_pc) ) == len(val_dataset_sketch)

    print(f"Experiment name [{opt.exp_name}]")

    exp_dir = Path(opt.models_dir, opt.exp_name)
    exp_dir.mkdir(exist_ok=True, parents=True)

    neptune_short_id = None
    neptune_short_id_file_path = exp_dir.joinpath('neptune_session.json')
    if neptune_short_id_file_path.is_file():
        with open(neptune_short_id_file_path, 'r') as neptune_short_id_file:
            try:
                neptune_session_json = json.load(neptune_short_id_file)
                if 'short_id' in neptune_session_json:
                    neptune_short_id = neptune_session_json['short_id']
                    print(f'Continuing Neptune run [{neptune_short_id}]')
            except:
                print("Could not resume neptune session")

    # create/load NeptuneLogger
    neptune_logger = None
    neptune_config_file_path = Path(__file__).parent.joinpath('..', 'config', 'neptune_config.yml').resolve()
    if neptune_config_file_path.is_file():
        print(f"Found neptune config file [{neptune_config_file_path}]")
        with open(neptune_config_file_path) as neptune_config_file:
            config = yaml.safe_load(neptune_config_file)
        api_token = config['neptune']['api_token']
        project = config['neptune']['project']
        tags = ["train"]
        if neptune_short_id:
            neptune_logger = NeptuneLogger( run=neptune.init(run=neptune_short_id, project=project, api_token=api_token, tags=tags), log_model_checkpoints=False )
        else:
            # log_model_checkpoints=False avoids saving the models to Neptune
            neptune_logger = NeptuneLogger(api_key=api_token, project=project, tags=tags, log_model_checkpoints=False)
        if neptune_short_id is None:
            # new experiment
            neptune_short_id = neptune_logger.run.fetch()['sys']['id']  # e.g. IN-105 (<project short identifier>-<run number>)
            with open(neptune_short_id_file_path, 'w') as neptune_short_id_file:
                json.dump({'short_id': neptune_short_id}, neptune_short_id_file)
                print(f'Started a new Neptune.ai run with id [{neptune_short_id}]')

    # log parameters to Neptune
    params = {
        "exp_name": opt.exp_name,
        "lr": 1e-2 if not opt.huang else 3e-4,
        "bs": opt.batch_size,
        "n_parameters": len(inputs_to_eval),
        "sched_step_size": 20,
        "sched_gamma": 0.85 if not opt.huang else 0.9,
        "normalize_embeddings": opt.normalize_embeddings,
        "increase_net_size": opt.increase_network_size,
        "pretrained_vgg": opt.pretrained_vgg,
        "use_regression": opt.use_regression,
    }
    if neptune_logger:
        neptune_logger.run['parameters'] = params

    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir,
        filename='ise-epoch{epoch:03d}-val_loss{val/loss/total:.2f}-val_acc{val/acc_top1/avg:.2f}',
        auto_insert_metric_name=False,
        save_last=True,
        monitor="val/acc_top1/avg",
        mode="max",
        save_top_k=3)

    huang_continuous = False
    huang_discrete = False
    if opt.huang == 'continuous':
        huang_continuous = True
    elif opt.huang == 'discrete':
        huang_discrete = True

    # import the relevant Model class
    if opt.huang:
        # comparison to Huang et al.
        from geocode_model_alexnet import Model
    else:
        from geocode_model import Model

    trainer = pl.Trainer(gpus=[0], max_epochs=opt.nepoch, logger=neptune_logger, callbacks=[checkpoint_callback])
    last_ckpt_file_name = f"{checkpoint_callback.CHECKPOINT_NAME_LAST}{checkpoint_callback.FILE_EXTENSION}"  # "last.ckpt" by default
    last_checkpoint_file_path = exp_dir.joinpath(last_ckpt_file_name)
    if last_checkpoint_file_path.is_file():
        print(f"Loading checkpoint file [{last_checkpoint_file_path}]...")
        pl_model = Model.load_from_checkpoint(str(last_checkpoint_file_path),
                                              param_descriptors=param_descriptors,
                                              trainer=trainer,
                                              models_dir=opt.models_dir,
                                              exp_name=opt.exp_name,
                                              use_regression=opt.use_regression,
                                              discrete=huang_discrete,
                                              continuous=huang_continuous)
    else:
        last_checkpoint_file_path = None
        if opt.huang:
            pl_model = Model(top_k_acc, opt.batch_size, detailed_vec_size, opt.increase_network_size, opt.normalize_embeddings, opt.pretrained_vgg,
                            opt.input_type, inputs_to_eval, params['lr'], params['sched_step_size'], params['sched_gamma'], opt.exp_name,
                            trainer=trainer, param_descriptors=param_descriptors, models_dir=opt.models_dir, use_regression=opt.use_regression,
                            discrete=huang_discrete, continuous=huang_continuous)
        else:
            # no huang related arguments
            pl_model = Model(top_k_acc, opt.batch_size, detailed_vec_size, opt.increase_network_size, opt.normalize_embeddings, opt.pretrained_vgg,
                            opt.input_type, inputs_to_eval, params['lr'], params['sched_step_size'], params['sched_gamma'], opt.exp_name,
                            trainer=trainer, param_descriptors=param_descriptors, models_dir=opt.models_dir, use_regression=opt.use_regression)
    trainer.fit(pl_model, train_dataloaders=combined_train_dataloader, val_dataloaders=combined_val_dataloader, ckpt_path=last_checkpoint_file_path)
