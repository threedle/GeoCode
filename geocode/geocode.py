#!/usr/bin/env python3

import argparse
from geocode_util import InputType
from geocode_train import train
from geocode_test import test


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected but got [{}].'.format(v))


def main():
    parser = argparse.ArgumentParser(prog='ShapeEditing')

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--dataset-dir', type=str, required=True, help='Path to dataset directory')
    common_parser.add_argument('--models-dir', type=str, required=True, help='Directory where experiments will be saved')
    common_parser.add_argument('--exp-name', type=str, required=True, help='Experiment directory within the models directory, where checkpoints will be saved')
    common_parser.add_argument('--input-type', type=InputType, nargs='+', default='pc sketch', help='Either \"pc\", \"sketch\" or \"pc sketch\"')
    common_parser.add_argument('--increase-network-size', action='store_true', default=False, help='Use larger encoders networks sizes')
    common_parser.add_argument('--normalize-embeddings', action='store_true', default=False, help='Normalize embeddings before using the decoders')
    common_parser.add_argument('--pretrained-vgg', action='store_true', default=False, help='Use a pretrained VGG network')
    common_parser.add_argument('--use-regression', action='store_true', default=False, help='Use regression instead of classification for continuous parameters')

    sp = parser.add_subparsers()
    sp_train = sp.add_parser('train', parents=[common_parser])
    sp_test = sp.add_parser('test', parents=[common_parser])

    sp_train.set_defaults(func=train)
    sp_test.set_defaults(func=test)

    sp_train.add_argument('--batch_size', type=int, required=True, help='Batch size')
    sp_train.add_argument('--nepoch', type=int, required=True, help='Number of epochs to train')

    sp_test.add_argument('--phase', type=str, default='test')
    sp_test.add_argument('--blender-exe', type=str, required=True, help='Path to blender executable')
    sp_test.add_argument('--blend-file', type=str, required=True, help='Path to blend file')
    sp_test.add_argument('--random-pc', type=int, default=None, help='Use only random point cloud sampling with specified number of points')
    sp_test.add_argument('--gaussian', type=float, default=0.0, help='Add Gaussian noise to the point cloud with the specified STD')
    sp_test.add_argument('--normalize-pc', action='store_true', default=False, help='Automatically normalize the input point clouds')
    sp_test.add_argument('--scanobjectnn', action='store_true', default=False, help='ScanObjectNN dataset which has only point clouds input')
    # we augment in phases "train", "val", "test" and experiments "coseg", "simplify_mesh", and "gaussian"
    # use `--augment-with-random-points false` to disable
    sp_test.add_argument('--augment-with-random-points', type=str2bool, default='True', help='Augment FPS point cloud with randomly sampled points')
    
    args = parser.parse_args()
    args.func(args)

    # either pc or sketch, or both must be trained
    assert InputType.pc in args.input_type or InputType.sketch in args.input_type


if __name__ == "__main__":
    main()
