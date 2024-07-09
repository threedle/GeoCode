#!/usr/bin/env python3

import zipfile
import hashlib
import requests
import traceback
from pathlib import Path
from argparse import ArgumentParser


def download_file(url, target_file_path):
    print(f"Downloading file from [{url}] as [{target_file_path}]")
    req = requests.get(url, allow_redirects=True)
    with open(target_file_path, 'wb') as target_file:
        target_file.write(req.content)


def download_ds(args):
    md5 = "b641562224202ff5afa86f023661e9c2"
    ds_url = "https://figshare.com/ndownloader/files/47453975?private_link=50549dabd53a72065749"
    ds_zip_file_name = "dataset_processing.zip"

    geocode_dir = Path('.').resolve()
    dataset_processing_dir_path = geocode_dir / "dataset_processing"
    dataset_processing_dir_path.mkdir(exist_ok=True)
    target_ds_processing_scripts_zip_file_path = dataset_processing_dir_path / ds_zip_file_name
    # download requested dataset processing scripts zip file from Google Drive
    if not target_ds_processing_scripts_zip_file_path.is_file():
        download_file(ds_url, target_ds_processing_scripts_zip_file_path)
    else:
        print(f"Skipping downloading dataset from Google Drive, file [{target_ds_processing_scripts_zip_file_path}] already exists.")

    unzipped_dataset_dir = target_ds_processing_scripts_zip_file_path.with_suffix('')

    if not unzipped_dataset_dir.is_dir():
        # verify md5
        print("Verifying MD5 hash...")
        assert hashlib.md5(open(target_ds_processing_scripts_zip_file_path, 'rb').read()).hexdigest() == md5

        print("Unzipping dataset...")
        with zipfile.ZipFile(target_ds_processing_scripts_zip_file_path, 'r') as target_ds_zip_file:
            target_ds_zip_file.extractall(dataset_processing_dir_path)
    else:
        print(f"Skipping dataset processing scripts unzip, directory [{unzipped_dataset_dir}] already exists.")

    print("Done")


def main():
    parser = ArgumentParser(prog='download_ds_processing_scripts')

    try:
        args = parser.parse_args()
        download_ds(args)
    except Exception as e:
        print(repr(e))
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
