import boto3
import argparse
import os 
import tarfile
import shutil
from parse_and_save_outputs import process_and_save_flux_data

def gather_directories(s3, bucket_name, subdirectory_prefix):
    """
    Lists objects within a specific "subdirectory" (prefix) in an S3 bucket.
    """
    paginator = s3.get_paginator('list_objects_v2')
    
    pages = paginator.paginate(Bucket=bucket_name, Prefix=subdirectory_prefix)
    
    tglfs = []
    tars = []
    for page in pages:
        if "Contents" in page:
            for obj in page['Contents']:
                fname = obj['Key']
                if '.tar' in fname:
                    tars.append(fname)
                    tglf_path = fname[:-31] + 'tglf/'
                    tglf_batch = []
                    for i in range(24):
                        idx = f'00{i}' if i < 10 else f'0{i}'
                        tglf_batch.append(tglf_path + f'input-{idx}/input.tglf')
                    tglfs.append(tglf_batch)
    return tglfs, tars

def download_data(s3, bucket_name, fname, out_path):
    key = fname.split('/')[-1]
    path = os.path.join(out_path, fname[:-len(key)])
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        s3.download_file(bucket_name, fname, os.path.join(path, key))
        print(f"File '{fname}' downloaded successfully to '{path}'")
        return os.path.join(path, key)
    except Exception as e:
        print(f"Error downloading file: {e}")

def untar(tar_path, prefix=""):
    # Create the extraction directory if it doesn't exist
    tar_key = tar_path.split('/')[-1]
    out = tar_path[:-len(tar_key)]
    out = os.path.join(out, prefix)
    os.makedirs(out, exist_ok=True)
    try:
        # Open the tar file in read mode
        with tarfile.open(tar_path, "r") as tar:
            # Extract all members (files and directories) to the specified path
            tar.extractall(path=out)
        print(f"Successfully extracted '{tar_path}' to '{out}'")
        return out

    except tarfile.ReadError:
        print(f"Error: Could not open or read '{tar_path}'. It might not be a valid tar file.")
    except FileNotFoundError:
        print(f"Error: The file '{tar_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def remove_dir(dir):
    try:
        shutil.rmtree(dir)
        print(f"Folder '{dir}' and its contents deleted successfully.")
    except OSError as e:
        print(f"Error deleting folder: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bucket")
    parser.add_argument("-d", "--directory")
    parser.add_argument("-o", "--output_path")

    args = parser.parse_args()
    bucket = args.bucket
    dir = args.directory
    out = args.output_path

    s3 = boto3.client(service_name='s3')

    print(f"\nObjects in {dir}:")
    tglfs, tars = gather_directories(s3, bucket, dir)
    for tglf_batch, tar in zip(tglfs, tars):
        # Download tar
        tar_path = download_data(s3, bucket, tar, out)
        # Download tglf inputs
        for tglf in tglf_batch:
            _ = download_data(s3, bucket, tglf, out)
        # Untar 
        untar_path = untar(tar_path, prefix='../../../../')
        # Process
        tar_key = tar.split('/')[-1]
        batch_name = tar_path.split('/')[5]
        job_name = tar_path.split('/')[4]
        job_path = tar_path[:-len(tar_key)] + '../../'
        process_and_save_flux_data(job_path, os.path.join(out, f'{job_name}_{batch_name}.h5'))
        # # Delete untarred dir
        remove_dir(job_path)