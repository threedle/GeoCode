#!/usr/bin/env python3

import gdown
import zipfile
import hashlib
import traceback
from pathlib import Path
from argparse import ArgumentParser


def download_ds(args):
    md5 = "41e5dd0df4ac5615bb2e24e977aaec22"
    ds_url = "https://drive.google.com/uc?id=1-NzLI--y1ewwZSX6kDJ3ALiC8UvolulR"
    ds_zip_file_name = "dataset_processing.zip"

    geocode_dir = Path('.').resolve()
    target_ds_processing_scripts_zip_file_path = geocode_dir.joinpath(ds_zip_file_name)
    # download requested dataset processing scripts zip file from Google Drive
    if not target_ds_processing_scripts_zip_file_path.is_file():
        gdown.download(ds_url, str(target_ds_processing_scripts_zip_file_path), quiet=False)
    else:
        print(f"Skipping downloading dataset from Google Drive, file [{target_ds_processing_scripts_zip_file_path}] already exists.")

    unzipped_dataset_dir = target_ds_processing_scripts_zip_file_path.with_suffix('')

    if not unzipped_dataset_dir.is_dir():
        # verify md5
        print("Verifying MD5 hash...")
        assert hashlib.md5(open(target_ds_processing_scripts_zip_file_path, 'rb').read()).hexdigest() == md5

        print("Unzipping dataset...")
        with zipfile.ZipFile(target_ds_processing_scripts_zip_file_path, 'r') as target_ds_zip_file:
            target_ds_zip_file.extractall(geocode_dir)
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
