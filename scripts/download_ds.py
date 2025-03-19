#!/usr/bin/env python3

import zipfile
import hashlib
import requests
import traceback
from pathlib import Path
from common.domain import Domain
from argparse import ArgumentParser


def download_file(url, target_file_path):
    print(f"Downloading file from [{url}] as [{target_file_path}]")
    req = requests.get(url, allow_redirects=True)
    with open(target_file_path, 'wb') as target_file:
        target_file.write(req.content)


def download_ds(args):
    datasets_dir = None
    if args.datasets_dir:
        datasets_dir = Path(args.datasets_dir)
        if not datasets_dir.is_dir():
            raise Exception(f'Given datasets path [{datasets_dir}] is not an existing directory.')
    models_dir = None
    if args.models_dir:
        models_dir = Path(args.models_dir)
        if not models_dir.is_dir():
            raise Exception(f'Given models path [{models_dir}] is not an existing directory.')
    blends_dir = None
    if args.blends_dir:
        blends_dir = Path(args.blends_dir)
        if not blends_dir.is_dir():
            raise Exception(f'Given blends path [{blends_dir}] is not an existing directory.')

    if args.domain == Domain.chair:
        md5 = "27c283fa6893b23400a9bba6aca92854"
        ds_url = "https://figshare.com/ndownloader/files/39487282?private_link=d06bff0ae6b0c710bec8"
        ds_zip_file_name = "ChairDataset.zip"
        best_epoch = 585
    elif args.domain == Domain.vase:
        md5 = "1200bfb9552513ea6c9a3b9050af470e"
        ds_url = "https://figshare.com/ndownloader/files/39487153?private_link=1b30f4105c0518ce9071"
        ds_zip_file_name = "VaseDataset.zip"
        best_epoch = 573
    elif args.domain == Domain.table:
        md5 = "c7a0fc73c2b3f39dcd02f8cd3380d9dd"
        ds_url = "https://figshare.com/ndownloader/files/39487033?private_link=53f9de1359c3e3cc3218"
        ds_zip_file_name = "TableDataset.zip"
        best_epoch = 537
    elif args.domain == Domain.ceiling_lamp:
        md5 = "a6e2e29790f74219539f4c151f566ba8"
        ds_url = "https://figshare.com/ndownloader/files/53115095?private_link=e568d4700d54a8f48289"
        ds_zip_file_name = "CeilingLampDataset.zip"
        best_epoch = 259
    else:
        raise Exception(f'Domain [{args.domain}] is not recognized.')

    if args.datasets_dir:
        target_ds_zip_file_path = datasets_dir.joinpath(ds_zip_file_name)
        # download requested dataset zip file from Google Drive
        if not target_ds_zip_file_path.is_file():
            download_file(ds_url, target_ds_zip_file_path)
        else:
            print(f"Skipping downloading dataset from Google Drive, file [{target_ds_zip_file_path}] already exists.")

        unzipped_dataset_dir = datasets_dir.joinpath(f"{str(args.domain).title()}Dataset")

        if not unzipped_dataset_dir.is_dir():
            # verify md5
            print("Verifying MD5 hash...")
            assert hashlib.md5(open(target_ds_zip_file_path, 'rb').read()).hexdigest() == md5

            print("Unzipping dataset...")
            with zipfile.ZipFile(target_ds_zip_file_path, 'r') as target_ds_zip_file:
                target_ds_zip_file.extractall(datasets_dir)
        else:
            print(f"Skipping dataset unzip, directory [{unzipped_dataset_dir}] already exists.")

    release_url = "https://github.com/threedle/GeoCode/releases/latest/download"

    if args.models_dir:
        best_ckpt_file_name = f"procedural_{args.domain}_last_ckpt.zip"
        latest_ckpt_file_name = f"procedural_{args.domain}_epoch{best_epoch:03d}_ckpt.zip"
        exp_target_dir = models_dir.joinpath(f"exp_geocode_{args.domain}")
        exp_target_dir.mkdir(exist_ok=True)

        best_ckpt_url = f"{release_url}/{best_ckpt_file_name}"
        best_ckpt_file_path = exp_target_dir.joinpath(best_ckpt_file_name)
        download_file(best_ckpt_url, best_ckpt_file_path)

        print(f"Unzipping checkpoint file [{best_ckpt_file_path}]...")
        with zipfile.ZipFile(best_ckpt_file_path, 'r') as best_ckpt_file:
            best_ckpt_file.extractall(exp_target_dir)

        latest_ckpt_url = f"{release_url}/{latest_ckpt_file_name}"
        latest_ckpt_file_path = exp_target_dir.joinpath(latest_ckpt_file_name)
        download_file(latest_ckpt_url, latest_ckpt_file_path)

        print(f"Unzipping checkpoint file [{latest_ckpt_file_path}]...")
        with zipfile.ZipFile(latest_ckpt_file_path, 'r') as latest_ckpt_file:
            latest_ckpt_file.extractall(exp_target_dir)

    if args.blends_dir:
        blend_file_name = f"procedural_{args.domain}.blend"
        blend_file_path = blends_dir.joinpath(blend_file_name)
        blend_url = f"{release_url}/{blend_file_name}"
        download_file(blend_url, blend_file_path)

    print("Done")


def main():
    parser = ArgumentParser(prog='download_ds')
    parser.add_argument('--domain', type=Domain, choices=list(Domain), required=True, help='The domain name to download the dataset for.')
    parser.add_argument('--datasets-dir', type=str, default=None, help='The directory to download the dataset to.')
    parser.add_argument('--models-dir', type=str, default=None, help='The directory to download checkpoint file to.')
    parser.add_argument('--blends-dir', type=str, default=None, help='The directory to download blend file to.')

    try:
        args = parser.parse_args()
        download_ds(args)
    except Exception as e:
        print(repr(e))
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
